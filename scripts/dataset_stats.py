from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name",type=str)
parser.add_argument("--dataset_path",type=str)
args = parser.parse_args()  

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
dataset = Dataset.load_from_disk(args.dataset_path)
print(dataset[0])

dataset = dataset.map(lambda l: {'q_len': len(tokenizer.tokenize(l['content']))})
dataset = dataset.map(lambda l: {'label_len': np.max([len(tokenizer.tokenize(lab)) for lab in l['label']])})
av_q_len_tok = round(np.mean(dataset['q_len']), 2)
av_max_label_len_tok = round(np.mean(dataset['label_len']), 2)
print('mean question length tokenized', av_q_len_tok)
print('mean label length tokenized', av_max_label_len_tok)
