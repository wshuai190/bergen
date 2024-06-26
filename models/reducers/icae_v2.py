from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from models.reducers.reducer import Reducer


class Splade(Reducer):
    def __init__(self, model_name):

        self.model_name = model_name
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()
        self.reverse_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        self.sep_token = self.tokenizer.sep_token_id
        self.device = "cuda"

    def __call__(self, kwargs):
        kwargs = {key: value.to('cuda') for key, value in kwargs.items() if (key != 'doc') and (key != 'd_id')}
        outputs = self.model(**kwargs).logits

        # pooling over hidden representations
        emb, _ = torch.max(torch.log(1 + torch.relu(outputs)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)

        return {
            "embedding": emb
        }

    def collate_fn(self, examples):
        query = [e['query'] for e in examples]
        doc = [e['doc'] for e in examples]
        d_id = [e['d_id'] for e in examples]
        query_reformed = []
        # query is a list of strings, d_ids are nested list, need to duplicate for each query wih respect to the lenth of doc in the query
        for i in range(len(query)):
            query_reformed += [query[i]] * len(doc[i])
        query = query_reformed

        # doc and d_ids are nested list, need to flat
        doc = [item for sublist in doc for item in sublist]
        d_id = [item for sublist in d_id for item in sublist]

        inp_dict = self.tokenizer(doc, padding=True, truncation="only_second", return_tensors='pt')
        inp_dict['d_id'] = d_id

        return inp_dict

    def reduce_fn(self, batch, topk=None):

        docs_tokenized = batch["input_ids"]

        for i in range(len(docs_tokenized)):
            docs_tokenized[i] = docs_tokenized[i].to("cuda")

        return docs_reduced, proportions