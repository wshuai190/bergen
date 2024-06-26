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
        doc = [e['doc'] for e in examples]
        d_id = [e['d_id'] for e in examples]
        # doc and d_ids are nested list, need to flat
        doc = [item for sublist in doc for item in sublist]
        d_id = [item for sublist in d_id for item in sublist]

        inp_dict = self.tokenizer(doc, padding=True, truncation="only_second", return_tensors='pt')
        inp_dict['d_id'] = d_id

        return inp_dict

    def reduce_fn(self, batch, topk=None):
        doc_embeds = self(batch)
        combineds_tokenized = batch["input_ids"]

        proportions = []
        batch_reduced_tokenized = []
        for i in range(combineds_tokenized.size(0)):
            doc_tokenized = combineds_tokenized[i].to("cuda")
            doc_embed = doc_embeds["embedding"][i].to("cuda")
            # Process embedding
            doc_rep = doc_embed.squeeze()
            col = torch.nonzero(doc_rep, as_tuple=True)[0]
            weights = doc_rep[col]
            sorted_indices = torch.argsort(weights, descending=True)
            if topk is not None:
                sorted_indices = sorted_indices[:topk]

            top_k = col[sorted_indices]
            reduced_tokenized = doc_tokenized[torch.isin(doc_tokenized, top_k)]
            batch_reduced_tokenized.append(reduced_tokenized)

            # Decode and calculate proportion
            proportion = len(reduced_tokenized) / len(doc_tokenized)
            proportions.append(proportion)

        max_length = max(len(t) for t in batch_reduced_tokenized)
        padded_reduced_tokenized = [torch.cat([t, torch.zeros(max_length - len(t), dtype=t.dtype, device="cuda")])
                                    for t in batch_reduced_tokenized]

        # Batch decode
        padded_reduced_tokenized = torch.stack(padded_reduced_tokenized)
        docs_reduced = self.tokenizer.batch_decode(padded_reduced_tokenized, skip_special_tokens=True)

        return docs_reduced, proportions