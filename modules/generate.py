'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
from torch.utils.data import DataLoader
from tqdm import tqdm
from hydra.utils import instantiate
# Generate
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from modules.dataset import Tokenized_Sorted_Dataset
from models.generators.llm_cocom import COCOMLLM
from models.generators.llm_icae import LLMICAE
from models.generators.oracle_provenance import OracleProvenance
from models.generators.oracle_answer import OracleAnswer
import torch


class Generate():
    def __init__(self, 
                 prompt=None,
                 init_args=None, 
                 batch_size=1,
                generation_top_k=1,
                 ):

        self.batch_size = batch_size
        # instatiate model
        self.model = instantiate(init_args, prompt=prompt, generation_top_k=generation_top_k)

    def eval(self, dataset):
        if isinstance(self.model, COCOMLLM) or isinstance(self.model, OracleProvenance) or isinstance(self.model, OracleAnswer) or isinstance(self.model, LLMICAE):
            dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                    collate_fn=lambda l: self.model.collate_fn(l, eval=True), num_workers=4)
        else:
            print("Not Using COCOMLLM")
            tokenized_and_sorted_dataset = Tokenized_Sorted_Dataset(dataset, self.model, training=False)
            dataloader = DataLoader(tokenized_and_sorted_dataset, batch_size=self.batch_size,
                                    collate_fn=lambda l: self.model.collate_fn(l, eval=True), num_workers=4)

        responses, instructions, query_ids, queries, labels, ranking_labels = list(), list(), list(), list(), list(), list()
        with torch.no_grad():
            for data_dict in tqdm(dataloader, desc='Generating'):
                id_ = data_dict['q_id']
                instruction = data_dict['instruction']
                query_ids += id_
                label = data_dict['label']
                labels += label
                queries += data_dict['query']
                ranking_labels += data_dict['ranking_label']
                instructions += instruction
                generated_response = self.model.generate(data_dict['model_input'])
                responses += generated_response
        return query_ids, queries, instructions, responses, labels, ranking_labels
