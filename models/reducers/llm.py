from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from models.reducers.reducer import Reducer

prompts = {
    "qd": {"system": "Answer the question in maximum one short sentence; if the question is unanswerable, write 'not provided'",
             "user": "Context: {context}\nQuestion: {question}",
             "system_without_docs": "Answer the question"},
    "d": {"system": "Summarize the context in maximum one short sentence but maintain all useful information for question answering",
             "user": "Context: {context}",
             "system_without_docs": "Summarize"}
}


class LLM(Reducer):
    def __init__(self, model_name, reduce_type="d", max_new_tokens=64, max_length=None):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length

        assert reduce_type in ["qd", "d"]
        self.reduce_type = reduce_type
        self.model.config.use_cache = True
        self.model = self.model.bfloat16()
        self.model.eval()
        self.sep_token = self.tokenizer.sep_token_id
        self.device = "cuda"

    def __call__(self, kwargs):
        return None

    def collate_fn(self, examples):
        queries = [e['query'] for e in examples]
        docs = [e['doc'] for e in examples]
        d_id = [e['d_id'] for e in examples]
        queries_reformed = []
        # query is a list of strings, d_ids are nested list, need to duplicate for each query wih respect to the lenth of doc in the query
        for i in range(len(queries)):
            queries_reformed += [queries[i]] * len(docs[i])
        queries = queries_reformed

        # doc and d_ids are nested list, need to flat
        docs = [item for sublist in docs for item in sublist]
        d_id = [item for sublist in d_id for item in sublist]
        docs_len = [len(doc.split()) for doc in docs]

        formatted_instructions = []
        for i in range(len(docs)):
            query = queries[i]
            doc = docs[i]
            formatted_instruction = self.format_instruction(query, doc)
            formatted_instructions.append(formatted_instruction)

        inp_dict = self.tokenizer(formatted_instructions, padding="longest", truncation=True, return_tensors='pt', max_length=self.max_length)

        inp_dict['d_id'] = d_id
        inp_dict['docs_len'] = docs_len

        return inp_dict

    def reduce_fn(self, batch, topk=None):
        instr_tokenized = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        output_ids = self.model.generate(
            instr_tokenized,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            max_length=self.max_length
        )

        prompt_len = instr_tokenized.size(1)
        generated_ids = output_ids[:, prompt_len:]

        docs_reduced = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        original_docs_len = batch["docs_len"]
        reduced_docs_len = [len(doc.split()) for doc in docs_reduced]
        proportions = [reduced_len / original_len for reduced_len, original_len in zip(reduced_docs_len, original_docs_len)]

        return docs_reduced, proportions

    def format_instruction(self, query, doc):
        prompt_format = prompts[self.reduce_type]
        if self.tokenizer.chat_template == None:
            formatted_prompt = prompt_format["system"] + "\n" + \
            prompt_format["user"].format(context=doc, question=query)
        else:
            formated_dict = [
                {"role": "system", "content": prompt_format["system"] },
                {"role": "user", "content": prompt_format["user"].format(context=doc, question=query)}]
            formatted_prompt = self.tokenizer.apply_chat_template(formated_dict, add_generation_prompt=True, tokenize=False)
        return formatted_prompt



