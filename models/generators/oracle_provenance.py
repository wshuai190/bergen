'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from models.generators.generator import Generator
from transformers import  AutoTokenizer

class OracleProvenance(Generator):
    def __init__(self, 
                 model_name=None, 
                 context_max_length=128,
                 test_mode="ft",
                 **kwargs
                 ):
        self.model_name = model_name
        self.pseudo_tokenizer=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=True)
        self.context_max_length = context_max_length
        self.test_mode = test_mode

    def tokenizer(self, instr, **kwargs):
        return instr

    def prediction_step(self, model, model_input, label_ids=None):
        pass

    def format_instruction(self, sample):
        docs_prompt = ''
        # for i, doc in enumerate(sample['doc']):
        #     docs_prompt += f"{doc} "
        doc = sample['doc'][0]
        # tokenize, then select the first max_new_tokens tokens
        doc_tokenized = self.pseudo_tokenizer(doc, truncation=True, max_length=self.context_max_length, return_tensors='pt', add_special_tokens=False)
        docs_prompt = self.pseudo_tokenizer.decode(doc_tokenized['input_ids'][0], skip_special_tokens=True)
        return f"""{docs_prompt}"""
    
    def generate(self, inp):
        return inp
        
    def collate_fn(self, examples, **kwargs):
        q_ids = [e['q_id'] for e in examples]
        instr = [self.format_instruction(e) for e in examples]
        label = [[e['label']] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)
        return {
            'model_input': instr,
            'q_id': q_ids, 
            'query': query, 
            'instruction': instr,
            'label': label, 
            'ranking_label': ranking_label,
        }