import re
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)
from utils import *



global autoais_model, autoais_tokenizer
autoais_model, autoais_tokenizer = None, None
QA_MODEL="gaotianyu1350/roberta-large-squad"
AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"

def make_group_prompt(group, group_id, group_prompt):
    # For doc prompt:
    # - {ID}: doc id (starting from 1)
    # - {T}: title
    # - {P}: text
    # use_shorter: None, "summary", or "extraction"

    text = group
    return group_prompt.replace("{P}", text).replace("{ID}", str(group_id+1))


def make_demo_groupgen(item, prompt, ndoc=None, doc_prompt=None, groupgen_instruction=None, 
                       group_prompt=None, use_shorter=None, 
              test=False):
    # For demo prompt
    # - {INST}: the instruction
    # - {D}: the documents
    # - {Q}: the question
    # - {G}: the groups
    # ndoc: number of documents to put in context
    # use_shorter: None, "summary", or "extraction"

    prompt = prompt.replace("{INST}", groupgen_instruction).replace("{Q}", item['question'])
    if "{D}" in prompt:
        if ndoc == 0:
            prompt = prompt.replace("{D}\n", "") # if there is no doc we also delete the empty line
        else:
            doc_list = get_shorter_text(item, item["docs"], ndoc, use_shorter) if use_shorter is not None else item["docs"][:ndoc]
            text = "".join([make_doc_prompt(doc, doc_id, doc_prompt, use_shorter=use_shorter) for doc_id, doc in enumerate(doc_list)])
            prompt = prompt.replace("{D}", text)
            
    if "{G}" in prompt:
        if not test:
            group_list = item["groups"]
            ngroup = len(item["groups"])
            if ngroup == 0:
                prompt = prompt.replace("{G}\n", "") # if there is no group we also delete the empty line
            else:
                text = "".join([make_group_prompt(group, group_id, group_prompt) \
                                for group_id, group in enumerate(group_list)])
                prompt = prompt.replace("{G}", text)
        else:
            prompt = prompt.replace("{G}", "").rstrip()

    return prompt

def make_demo_grouppred(item, prompt, grouppred_instruction=None, group_prompt=None, test=False):
    # For demo prompt
    # - {INST}: the instruction
    # - {G}: the groups
    # - {Q}: the question
    # - {A}: the answers
    # ndoc: number of documents to put in context

    prompt = prompt.replace("{INST}", grouppred_instruction).replace("{Q}", item['question'])
    
    if "{G}" in prompt:
        # prompt = prompt.replace("{G}", item["groups_str"])
        group_list = item["groups"]
        ngroup = len(item["groups"])
        if ngroup == 0:
            prompt = prompt.replace("{G}\n", "") # if there is no group we also delete the empty line
        else:
            text = "".join([make_group_prompt(group, group_id, group_prompt) \
                            for group_id, group in enumerate(group_list)])
            prompt = prompt.replace("{G}", text)
            
    if not test:
        answer = "\n" + "\n".join(item["answer_from_groups"]) if isinstance(item["answer_from_groups"], list) \
                else item["answer_from_groups"]
        prompt = prompt.replace("{A}", "").rstrip() + answer
    else:
        prompt = prompt.replace("{A}", "").rstrip() # remove any space or \n

    return prompt

# retrieval
def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

def find_external_docs_idx(question, external):
    # find the index of externaol doc that matches the question
    for idx, item in enumerate(external):
        if item['question'] == question:
            return idx
    return None

def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference

def compute_autoais_single(sent, docs):
    """
    Compute AutoAIS score.

    Args:
        sent: str, one sentence
        docs: list of str, multiple sentences
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        logger.info("Loading AutoAIS model...")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    logger.info(f"Running AutoAIS...")

    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc['title'], doc['sent'])
        else:
            return "Title: %s\n%s" % (doc['title'], doc['text'])
        
    sent = remove_citations(sent).strip()
    target_sent = sent
    joint_passage = '\n'.join([_format_document(doc) for doc in docs])
    joint_entail = _run_nli_autoais(joint_passage, target_sent)
    entail = joint_entail
    # calculate the precision score if applicable
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    entail_prec = 0
    if joint_entail and len(docs) > 1:
        sent_mcite_support += 1
        # Precision check: did the model cite any unnecessary documents?
        for psgs_id in range(len(docs)):
            # condition A
            passage = _format_document(docs[psgs_id]) 
            nli_result = _run_nli_autoais(passage, target_sent)

            # condition B
            if not nli_result:
                subset_exclude = docs[:psgs_id] + docs[psgs_id + 1:]
                passage = '\n'.join([_format_document(doc) for doc in subset_exclude])
                nli_result = _run_nli_autoais(passage, target_sent)
                if nli_result: # psgs_id is not necessary
                    flag = 0
                    sent_mcite_overcite += 1 
                else:
                    entail_prec += 1
            else:
                entail_prec += 1
    else:
        entail_prec += joint_entail 
    total_citations = len(docs)
    
    return {
        'entail': entail,
        'entail_prec': entail_prec / total_citations
    }