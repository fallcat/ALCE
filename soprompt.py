import os
import openai
import json
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import time
import string
import numpy as np
import torch
import re
import sys
# sys.path.append('..')
from searcher import SearcherWithinDocs
import yaml
from utils import *
import nltk
nltk.download('punkt')
from nltk import sent_tokenize
from run import LLM
from soprompt_utils import *

import os
import torch
import numpy as np
import copy
# from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import namedtuple, defaultdict


SOPROMPT_DOC_MODES = ['retrieved', 'postcited', 'comb', 'none']
SOPROMPT_PRED_MODES = ['docs', 'groups']

class SoPrompt():
    def __init__(self, 
                 llm,
                 args,
                 prompt_data,
                 get_groupgen_prompt=None,
                 get_grouppred_prompt=None,
                 get_prompt=None,
                 get_closedbook_prompt=None
                ):
        
        self.llm = llm
        self.args = args
        self.prompt_data = prompt_data

        self.init_retriever()
        self.init_prompt_getters(
            get_groupgen_prompt=get_groupgen_prompt,
            get_grouppred_prompt=get_grouppred_prompt,
            get_prompt=get_prompt,
            get_closedbook_prompt=get_closedbook_prompt
        )
        self.init_head_prompts()

    def init_retriever(self):
        # Load retrieval model
        if "gtr" in self.args.retriever:
            from sentence_transformers import SentenceTransformer
            self.gtr_model = SentenceTransformer(f'sentence-transformers/{self.args.retriever}', 
                                            device=self.args.retriever_device)
        else:
            self.gtr_model = None

    def init_prompt_getters(self, 
        get_groupgen_prompt=None,
        get_grouppred_prompt=None,
        get_prompt=None,
        get_closedbook_prompt=None
    ):
        ndoc = self.args.ndoc
        if self.args.no_doc_in_demo:
            ndoc = 0
        elif self.args.fewer_doc_in_demo:
            assert self.args.ndoc_in_demo is not None
            ndoc = self.args.ndoc_in_demo
        self.ndoc = ndoc
        if get_groupgen_prompt is None:
            def get_groupgen_prompt(eval_item, test=False):
                groupgen_prompt = make_demo_groupgen(
                    eval_item, prompt=self.prompt_data["groupgen_prompt"], ndoc=ndoc, 
                    doc_prompt=self.prompt_data["doc_prompt"],
                    groupgen_instruction=self.prompt_data["groupgen_instruction"], 
                    group_prompt=self.prompt_data["group_prompt"],
                    test=test
                )
                return groupgen_prompt
        self.get_groupgen_prompt = get_groupgen_prompt
        
        if get_grouppred_prompt is None:
            def get_grouppred_prompt(eval_item, test=False):
                grouppred_prompt = make_demo_grouppred(
                    eval_item, prompt=self.prompt_data["grouppred_prompt"], 
                    grouppred_instruction=self.prompt_data["grouppred_instruction"], 
                    group_prompt=self.prompt_data["group_prompt"],
                    test=test
                )
                return grouppred_prompt
        self.get_grouppred_prompt = get_grouppred_prompt
        
        if get_prompt is None:
            def get_prompt(eval_item, test=False):
                prompt = make_demo(
                    eval_item, prompt=self.prompt_data["demo_prompt"], ndoc=ndoc, 
                    doc_prompt=self.prompt_data["doc_prompt"], 
                    instruction=self.prompt_data["instruction"], use_shorter=self.args.use_shorter, 
                    test=test
                )
                return prompt
        self.get_prompt = get_prompt
        
        if get_closedbook_prompt is None:
            def get_closedbook_prompt(eval_item, test=False):
                prompt = make_demo(
                    eval_item, prompt=self.prompt_data["closedbook_prompt"], ndoc=0, 
                    doc_prompt=self.prompt_data["doc_prompt"], 
                    instruction=self.prompt_data["instruction"], use_shorter=self.args.use_shorter, 
                    test=test
                )
                return prompt
        self.get_closedbook_prompt = get_closedbook_prompt
        
    def init_head_prompts(self):
        # Generate the demonstration part
        head_prompt, head_groupgen_prompt, head_grouppred_prompt, head_closedbook_prompt = "", "", "", ""
        train_ids = np.random.choice(len(self.prompt_data["demos"]), self.args.shot, replace=False)
        for train_id in train_ids:
            train_item = self.prompt_data["demos"][train_id]
            ndoc = self.args.ndoc
            if self.args.no_doc_in_demo:
                ndoc = 0
            elif self.args.fewer_doc_in_demo:
                assert self.args.ndoc_in_demo is not None
                ndoc = self.args.ndoc_in_demo
            head_prompt += self.get_prompt(train_item) + self.prompt_data["demo_sep"]
            head_groupgen_prompt += self.get_groupgen_prompt(train_item) + self.prompt_data["demo_sep"]
            head_grouppred_prompt += self.get_grouppred_prompt(train_item) + self.prompt_data["demo_sep"]
            head_closedbook_prompt += self.get_closedbook_prompt(train_item) + self.prompt_data["demo_sep"]
        self.head_prompt = head_prompt
        self.head_groupgen_prompt = head_groupgen_prompt
        self.head_grouppred_prompt = head_grouppred_prompt
        self.head_closedbook_prompt = head_closedbook_prompt
    
    def generate(self, data, doc_mode='retrieved', pred_mode='docs', 
                 return_dict=False, external=None):
        assert doc_mode in SOPROMPT_DOC_MODES
        assert pred_mode in SOPROMPT_PRED_MODES
        data = copy.deepcopy(data)
        data['doc_mode'] = doc_mode
        data['pred_mode'] = pred_mode
        data['ndoc'] = self.ndoc

        # DOCUMENT
        # What document to use? Retrieved, or ones generated by post-hoc citation? 
        # Or post-hoc citation that has the best entailment recall score appended to retrieved?
        if doc_mode in ['postcited', 'comb', 'none']:
            closedbook_output = self.closedbook_predict(data)
            closedbook_output['closedbook_generation'] = closedbook_output['generation']
            data.update(closedbook_output)
            results = self.add_postcite(closedbook_output['generation'], 
                                        data['question'],
                                        external)
            data.update(results)
            if doc_mode == 'none':
                data['generation'] = results['postcite_generation']
                if return_dict:
                    return data
                else:
                    return data['generation']
            else: # postcited
                if doc_mode == 'postcited':
                    data['docs'] = data['postcite_best_docs']
                else:
                    result = self.select_postcite(data)
                    data.update(result)
            
        # PREDICTION
        # What to predict on? From the documents, or from the groups?
        if pred_mode == 'docs':
            results = self.backbone_predict(data)
            data.update(results)
        elif pred_mode == 'groups':
            data = self.groupgen_pred(data)
        else:
            raise ValueError(f"pred_mode {pred_mode} not recognized")

        # Return the results
        if return_dict:
            return data
        else:
            return data['generation']

    def select_postcite(self, data):
        sents = data['postcite_sents']
        docs = data['postcite_best_docs']
        results = {}
        ais_results = defaultdict(list)
        docs_entail = []
        for sent, doc in zip(sents, docs):
            ais_results_single = compute_autoais_single(sent, [doc])
            # import pdb; pdb.set_trace()
            ais_results['entail'].append(ais_results_single['entail'])
            ais_results['entail_prec'].append(ais_results_single['entail_prec'])
            if ais_results_single['entail'] > 0:
                docs_entail.append(doc)
        results['original_docs'] = copy.deepcopy(data['docs'])
        results['postcite_docs'] = copy.deepcopy(docs_entail)
        new_docs = docs_entail
        num_new_used = len(docs_entail)
        num_postcite_overlap = 0
        for di, doc in enumerate(data['docs']):
            if doc['text'] not in [doc_['text'] for doc_ in new_docs]:
                new_docs.append(doc)
            else:
                if di < self.ndoc:
                    num_postcite_overlap += 1
        results['docs'] = new_docs # First put the docs that are retrieved. #docs_entail + data['docs']
        results['entail'] = ais_results['entail']
        results['entail_prec'] = ais_results['entail_prec']
        results['num_new_used'] = num_new_used
        results['num_postcite_overlap'] = num_postcite_overlap
        return results
    
    def add_postcite(self, prev_generation, question, external):
        external_idx = find_external_docs_idx(question, external)
        doc_list = external[external_idx]['docs']
        searcher = SearcherWithinDocs(doc_list, self.args.retriever, 
                                      model=self.gtr_model, 
                                      device=self.args.retriever_device)
        output = prev_generation.strip().split("\n")[0] # Remove new lines and content after
        output = prev_generation.replace("<|im_end|>", "")
        if "qampari" in self.args.dataset_name:
            sents = [question + ' ' + x.strip() 
                     for x in prev_generation.rstrip(".").split(",")]
        else:
            sents = sent_tokenize(output)

        new_output = ""
        best_doc_ids = []
        for sent in sents:
            original_ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] 

            if len(original_ref) == 0 or self.args.overwrite:
                sent = remove_citations(sent)
                best_doc_id = searcher.search(sent)
                sent = f"[{best_doc_id+1}] " + sent
                best_doc_ids.append(best_doc_id)

            if "qampari" in self.args.dataset_name:
                new_output += sent.replace(question, '').strip() + ", "
            else:
                new_output += sent + " "

        closedbook_phc_output = new_output.rstrip().rstrip(",")
        generation = closedbook_phc_output
        return {
            'postcite_sents': sents,
            'postcite_generation': generation,
            'postcite_best_doc_ids': best_doc_ids,
            'postcite_best_docs': [
                external[external_idx]['docs'][best_doc_id] 
                for best_doc_id in best_doc_ids
                ]
        }
        return generation
    
    def llm_predict(self, prompt):
        prompt_len = len(self.llm.tokenizer.tokenize(prompt))
        generation = self.llm.generate(prompt, min(self.args.max_new_tokens, 
                                                   self.args.max_length-prompt_len))
        return {
            'generation': generation,
            'prompt': prompt,
            'prompt_len': prompt_len
        }

    def closedbook_predict(self, data):
        prompt = self.head_closedbook_prompt + self.get_closedbook_prompt(data, test=True)
        results = self.llm_predict(prompt)
        return results
    
    def backbone_predict(self, data):
        prompt = self.head_prompt + self.get_prompt(data, test=True)
        results = self.llm_predict(prompt)
        return results
    
    def group_gen(self, data):
        groupgen_prompt = self.head_groupgen_prompt + self.get_groupgen_prompt(data, test=True)
        results = self.llm_predict(groupgen_prompt)
        groups_str = results['generation']
        groups = [':'.join(group.split(':')[1:]) for group in groups_str.split('\n')]
        results['groups'] = groups
        return results

    def group_predict(self, data):
        grouppred_prompt = self.head_grouppred_prompt + self.get_grouppred_prompt(data, test=True)
        results = self.llm_predict(grouppred_prompt)
        return results

    def groupgen_pred(self, data):
        groupgen_output = self.group_gen(data)
        groups = groupgen_output['groups']
        data['groups'] = groupgen_output['groups']
        data['groups_str'] = groupgen_output['generation']
        data['groupgen_prompt'] = groupgen_output['prompt']
        data['groupgen_prompt_len'] = groupgen_output['prompt_len']
        grouppred_output = self.group_predict(data)
        data['generation'] = grouppred_output['generation']
        data['grouppred_prompt'] = grouppred_output['prompt']
        data['grouppred_prompt_len'] = grouppred_output['prompt_len']
        return data