'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from modules.retrieve import Retrieve
from modules.rerank import Rerank
from modules.generate import Generate
from modules.dataset_processor import ProcessDatasets
from modules.metrics import RAGMetrics
from models.generators.llm_cocom import COCOMLLM
import time 
import shutil
import os 
from tqdm import tqdm
from hydra.utils import instantiate
from utils import (
    eval_retrieval_kilt, init_experiment, move_finished_experiment,
    write_trec, prepare_dataset_from_ids, load_trec,
    print_generate_out, print_rag_model,
    write_generated, write_dict, get_by_id, get_index_path, 
    get_reranking_filename, format_time, get_ranking_filename, get_finished_experiment_name
)

class RAG:
    def __init__(self, 
                generator=None, 
                retriever=None, 
                reranker=None,
                
                runs_folder=None,
                run_name=None, 
                dataset=None, 
                processing_num_proc=1,
                dataset_folder='datasets/',
                index_folder='indexes/',
                experiments_folder='experiments/', 
                qrels_folder='qrels/',
                overwrite_datasets=False,
                overwrite_exp=False,
                overwrite_index=False,
                retrieve_top_k=1,
                rerank_top_k=1,
                generation_top_k=1,
                pyserini_num_threads=1,
                config=None,
                debug=False,
                continue_batch=None,
                train=None,
                prompt=None,
                **kwargs,
                ):
        
        retriever_config = retriever
        reranker_config = reranker
        generator_config = generator
        dataset_config = dataset


        #if all the config are still None, load from config

        #if none, then load from config
        if generator_config == None:
            generator_config = config.generator if hasattr(config, 'generator') else None
        if retriever_config == None:
            retriever_config = config.retriever if hasattr(config, 'retriever') else None
        if reranker_config == None:
            reranker_config = config.reranker if hasattr(config, 'reranker') else None
        if dataset_config == None:
            dataset_config = config.dataset if hasattr(config, 'dataset') else None
        
        self.debug = debug
        self.dataset_folder = dataset_folder
        self.experiments_folder = experiments_folder
        self.runs_folder = runs_folder
        self.qrels_folder = qrels_folder
        self.run_name = run_name
        self.processing_num_proc = processing_num_proc
        self.index_folder = index_folder
        self.config = config
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.generation_top_k = generation_top_k
        self.pyserini_num_threads = pyserini_num_threads
        self.overwrite_exp = overwrite_exp
        self.overwrite_index = overwrite_index
        self.training_config = train

        assert self.generation_top_k <= self.rerank_top_k <= self.retrieve_top_k
        # init experiment (set run name, create dirs)
        self.run_name, self.experiment_folder = init_experiment(config, experiments_folder, index_folder, runs_folder, run_name, overwrite_exp=self.overwrite_exp, continue_batch=continue_batch)
        # process datasets, downloading, loading, covert to format
        self.datasets = ProcessDatasets.process(
            dataset_config, 
            out_folder=self.dataset_folder, 
            num_proc=processing_num_proc,
            overwrite=overwrite_datasets,
            debug=debug,
            shuffle_labels=True if generator_config != None and generator_config.init_args.model_name == 'random_answer' else False,
            oracle_provenance=True if retriever_config != None and retriever_config.init_args.model_name == 'oracle_provenance' else False,
            )
        
        self.metrics = {
            "train": RAGMetrics,
            # lookup metric with dataset name (tuple: dataset_name, split) 
            "dev": RAGMetrics, 
            "test": None,
        }

        # init retriever
        self.retriever = Retrieve(
                    **retriever_config,
                    pyserini_num_threads=self.pyserini_num_threads,
                    continue_batch=continue_batch,
                    ) if retriever_config != None else None
        # init reranker
        self.reranker = Rerank(
            **reranker_config,
            ) if reranker_config != None else None


        self.generator = Generate(**generator_config, prompt=prompt, generation_top_k=self.generation_top_k) if generator_config != None else None

        
                # print RAG model
        print_rag_model(self, retriever_config, reranker_config, generator_config)

        
    def eval(self, dataset_split):
        dataset = self.datasets[dataset_split]
        query_dataset_name = self.datasets[dataset_split]['query'].name
        doc_dataset_name = self.datasets[dataset_split]['doc'].name

        # retrieve
        if self.retriever != None:
            query_ids, doc_ids, _ = self.retrieve(
                    dataset, 
                    query_dataset_name, 
                    doc_dataset_name,
                    dataset_split, 
                    self.retrieve_top_k,
                    )  
        else:
            query_ids, doc_ids = None, None
        # rerank
        if self.reranker !=  None:
            query_ids, doc_ids, _ = self.rerank(
                dataset, 
                query_dataset_name, 
                doc_dataset_name,
                dataset_split, 
                query_ids, 
                doc_ids,
                self.rerank_top_k,
                )

        # generate
        if self.generator !=  None:
            questions, _, predictions, references = self.generate(
                dataset, 
                dataset_split,
                query_ids, 
                doc_ids
                )
            # eval metrics
            self.eval_metrics(
                dataset_split, 
                questions, 
                predictions, 
                references
                )

        move_finished_experiment(self.experiment_folder)


    def retrieve(self, 
                 dataset, 
                 query_dataset_name, 
                 doc_dataset_name,
                 dataset_split, 
                 retrieve_top_k,
                 eval_ranking=False,
                 ):
        
        ranking_file = get_ranking_filename(
            self.runs_folder,
            query_dataset_name,
            doc_dataset_name,
            self.retriever.get_clean_model_name(),
            dataset_split, 
            retrieve_top_k,

        )
        #if return_embeddings:
                #raise NotImplementedError('For returning Embeddings is not yet fully implemented!')
        doc_embeds_path = get_index_path(self.index_folder, doc_dataset_name, self.retriever.get_clean_model_name(), 'doc')
        query_embeds_path = get_index_path(self.index_folder, query_dataset_name, self.retriever.get_clean_model_name(), 'query', dataset_split=dataset_split)
        if not os.path.exists(ranking_file) or self.overwrite_exp or self.overwrite_index:
            print(f'Run {ranking_file} does not exists, running retrieve...')
             # retrieve
            out_ranking = self.retriever.retrieve(
                dataset,
                query_embeds_path,
                doc_embeds_path,
                retrieve_top_k,
                overwrite_index=self.overwrite_index
                )
            query_ids, doc_ids, scores = out_ranking['q_id'], out_ranking['doc_id'], out_ranking['score']
            write_trec(ranking_file, query_ids, doc_ids, scores)
        else:             
            query_ids, doc_ids, scores = load_trec(ranking_file)
        # copy ranking file to experiment folder    
        shutil.copyfile(ranking_file, f'{self.experiment_folder}/{ranking_file.split("/")[-1]}')
        if eval_ranking:
            if 'ranking_label' in self.datasets[dataset_split]['query'].features:
                print('Evaluating retrieval...')
                wiki_doc_ids = [get_by_id(self.datasets[dataset_split]['doc'], doc_ids_q, 'wikipedia_id') for doc_ids_q in tqdm(doc_ids, desc='Getting wiki ids...')]
                eval_retrieval_kilt(
                    self.experiment_folder, 
                    self.qrels_folder, 
                    query_dataset_name, 
                    dataset_split, query_ids, 
                    wiki_doc_ids, scores, 
                    top_k=self.generation_top_k, 
                    debug=self.debug,
                    )
        return query_ids, doc_ids, scores

    def rerank(self, 
               dataset, 
               query_dataset_name, 
               doc_dataset_name, 
               dataset_split, 
               query_ids, 
               doc_ids, 
               rerank_top_k, 
               return_embeddings=False,
               eval_ranking=False
               ):
        
        doc_ids = [doc_ids_q[:rerank_top_k] for doc_ids_q in doc_ids]

        reranking_file = get_reranking_filename(
            self.runs_folder,
            query_dataset_name,
            doc_dataset_name,
            dataset_split,
            self.retriever.get_clean_model_name(),
            self.retrieve_top_k,
            self.reranker.get_clean_model_name(),
            self.rerank_top_k,
        )

        if not os.path.exists(reranking_file) or self.overwrite_exp:
            rerank_dataset = prepare_dataset_from_ids(
                    dataset, 
                    query_ids, 
                    doc_ids,
                    multi_doc=False,
                )
            out_ranking = self.reranker.eval(rerank_dataset, return_embeddings=return_embeddings)
            query_ids, doc_ids, scores = out_ranking['q_id'], out_ranking['doc_id'], out_ranking['score']
            write_trec(reranking_file, query_ids, doc_ids, scores)
        else:
            # copy reranking file to experiment folder 
            shutil.copyfile(reranking_file, f'{self.experiment_folder}/{reranking_file.split("/")[-1]}')
            query_ids, doc_ids, scores = load_trec(reranking_file)
        if eval_ranking:
            if 'ranking_label' in self.datasets[dataset_split]['query'].features:
                print('Evaluating retrieval...')
                wiki_doc_ids = [get_by_id(dataset['doc'], doc_ids_q, 'wikipedia_id') for doc_ids_q in doc_ids]
                eval_retrieval_kilt(
                    self.experiment_folder, 
                    self.qrels_folder, 
                    query_dataset_name, 
                    dataset_split, 
                    query_ids, 
                    wiki_doc_ids, 
                    scores, 
                    top_k=self.generation_top_k, 
                    reranking=True, 
                    debug=self.debug
                    )
        return query_ids, doc_ids, scores


    def generate(self, 
                 dataset, 
                 dataset_split, 
                 query_ids, 
                 doc_ids,
                 ):
        doc_ids = [doc_ids_q[:self.generation_top_k] for doc_ids_q in doc_ids] if doc_ids != None else doc_ids 

        gen_dataset = prepare_dataset_from_ids(
            dataset, 
            query_ids, 
            doc_ids,
            multi_doc=True, 
            )

        generation_start = time.time()
        query_ids, questions, instructions, predictions, references, ranking_labels  = self.generator.eval(gen_dataset)
        generation_time = time.time() - generation_start
        write_generated(
            self.experiment_folder,
            f"eval_{dataset_split}_out.json",
            query_ids, 
            questions,
            instructions, 
            predictions, 
            references, 
            ranking_labels
        )

        print_generate_out(
            questions,
            instructions,
            predictions,
            query_ids, 
            references,
            ranking_labels,
            )

        
        if hasattr(self.generator.model,"total_cost"):
            print(self.generator.model.total_cost,self.generator.model.prompt_cost, self.generator.model.completion_cost)
            write_dict(self.experiment_folder, f"eval_{dataset_split}_generation_cost.json", 
                       {'total_cost':self.generator.model.total_cost,
                        'prompt_cost':self.generator.model.prompt_cost,
                        'completion_cost':self.generator.model.completion_cost}
                        )


        formated_time_dict = format_time("Generation time", generation_time)
        write_dict(self.experiment_folder, f"eval_{dataset_split}_generation_time.json", formated_time_dict)

        return questions, instructions, predictions, references

    def eval_metrics(self, dataset_split, questions, predictions, references):
        if predictions == references == questions == None:
            return
        metrics_out = self.metrics[dataset_split].compute(
        predictions=predictions, 
        references=references, 
        questions=questions
        )
        write_dict(self.experiment_folder, f"eval_{dataset_split}_metrics.json", metrics_out)
    

    def train(self):
        from transformers import TrainingArguments
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from modules.trainer import RAGTrainer
        import torch
        from transformers import Trainer
        from modules.dataset import Tokenized_Sorted_Dataset
        from torch.utils.data import DataLoader
        from accelerate import Accelerator
        from utils import set_seed
        accelerator = Accelerator()

        
        dataset_split = 'train'
        dataset = self.datasets[dataset_split] 
        query_dataset_name = dataset['query'].name
        doc_dataset_name = dataset['doc'].name

        # if no retriever don't load doc embeddings
        if self.retriever != None:
            query_ids, doc_ids, _ = self.retrieve(
                dataset, 
                query_dataset_name, 
                doc_dataset_name,
                dataset_split, 
                self.retrieve_top_k,
                eval_ranking=False
                )            
        else:
            query_ids, doc_ids = None, None

        if self.reranker !=  None:
            query_ids, doc_ids, _ = self.rerank(
                dataset,
                query_dataset_name,
                doc_dataset_name,
                dataset_split,
                query_ids,
                doc_ids,
                self.rerank_top_k,
                )

        # get top-k docs
        doc_ids = [doc_ids_q[:self.generation_top_k] for doc_ids_q in doc_ids] if doc_ids != None else doc_ids

        query_ids = query_ids
        doc_ids = doc_ids

        # prepare dataset
        gen_dataset = prepare_dataset_from_ids(
            dataset, 
            query_ids, 
            doc_ids, 
            multi_doc=True, 
            )
        # split train into train and test
        train_test_datasets = gen_dataset.train_test_split(self.training_config.test_size_ratio, seed=42)
        if isinstance(self.generator.model, COCOMLLM):
            print("Preprocessing data...")
            #call_back_data_select = DataLoader(train_test_datasets['test'].select(range(self.training_config.generate_test_samples)), batch_size=self.training_config.trainer.per_device_eval_batch_size, collate_fn=lambda l: self.generator.model.collate_fn(l, eval=True))
        else:
            print("Preprocessing data...")
            train_test_datasets['train'] = Tokenized_Sorted_Dataset(train_test_datasets['train'], self.generator.model, training=True)
            train_test_datasets['test'] = Tokenized_Sorted_Dataset(train_test_datasets['test'], self.generator.model, training=False)
            call_back_data = Tokenized_Sorted_Dataset(train_test_datasets['test'], self.generator.model, training=False)
            call_back_data_select = DataLoader(call_back_data.select(range(self.training_config.generate_test_samples)), batch_size=self.training_config.trainer.per_device_eval_batch_size, collate_fn=lambda l: self.generator.model.collate_fn(l, eval=True))

        print("Data preprocessed")

        # if lora in train config
        if 'lora' in self.training_config:
            self.generator.model.model = prepare_model_for_kbit_training(self.generator.model.model)
            print("using lora training")
            # lora config
            lora_config = LoraConfig(
                **self.training_config.lora,
                target_modules=['q_proj', 'down_proj', 'gate_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj'],
                )
            # get adapter
            self.generator.model.model = get_peft_model(self.generator.model.model, lora_config)
            self.generator.model.model.print_trainable_parameters()

        total_batch_size = self.training_config.trainer.per_device_train_batch_size * torch.cuda.device_count() * self.training_config.trainer.gradient_accumulation_steps
        total_steps = len(train_test_datasets['train']) * self.training_config.trainer.num_train_epochs // total_batch_size
        num_saving_steps = 2
        eval_steps =  max(total_steps// num_saving_steps, 1)
        save_steps = max(total_steps  // num_saving_steps, 1)
        logging_steps = max((total_steps // num_saving_steps) // 2, 1)


        args = TrainingArguments(
            run_name=self.run_name,
            output_dir=f'{self.experiment_folder}/train/',
            **self.training_config.trainer,
            evaluation_strategy="steps",
            eval_steps=eval_steps, 
            save_steps=save_steps,
            logging_steps=logging_steps,
            load_best_model_at_end=True,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        )

        
        set_seed(42)
        trainer = RAGTrainer(
            model=self.generator.model.model,
            model_prediction_step=self.generator.model.prediction_step,
            generate=self.generator.model.generate,
            args=args,
            data_collator=self.generator.model.collate_fn,
            train_dataset=train_test_datasets['train'],
            eval_dataset=train_test_datasets['test'],
        )
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(trainer.model, trainer.optimizer,
                                                                              trainer.get_train_dataloader(),
                                                                              trainer.get_eval_dataloader())

        trainer.train()
        self.generator.model.model = trainer.model

        if accelerator.is_main_process:
            move_finished_experiment(self.experiment_folder)
        self.experiment_folder = get_finished_experiment_name(self.experiment_folder)




    # def cocom_tune(self):
    #     from transformers import TrainingArguments
    #     import torch
    #     from transformers import Trainer

    #     dataset_split = 'train'
    #     dataset = self.datasets[dataset_split]
    #     query_dataset_name = dataset['query'].name
    #     doc_dataset_name = dataset['doc'].name

    #     # if no retriever don't load doc embeddings
    #     if self.retriever != None:
    #         query_ids, doc_ids, _ = self.retrieve(
    #             dataset,
    #             query_dataset_name,
    #             doc_dataset_name,
    #             dataset_split,
    #             self.retrieve_top_k,
    #             eval_ranking=False
    #         )
    #     else:
    #         query_ids, doc_ids = None, None

    #     if self.reranker != None:
    #         query_ids, doc_ids, _ = self.rerank(
    #             dataset,
    #             query_dataset_name,
    #             doc_dataset_name,
    #             dataset_split,
    #             query_ids,
    #             doc_ids,
    #             self.rerank_top_k,
    #         )
    #     gen_dataset = prepare_dataset_from_ids(
    #         dataset,
    #         query_ids,
    #         doc_ids,
    #         multi_doc=True,
    #     )
    #     # split train into train and test
    #     train_test_datasets = gen_dataset.train_test_split(self.training_config.test_size_ratio, seed=42)

    #     total_batch_size = self.training_config.trainer.per_device_train_batch_size * torch.cuda.device_count()
    #     total_steps = len(train_test_datasets['train']) // total_batch_size
    #     num_saving_steps = 5
    #     eval_steps = max(total_steps // num_saving_steps, 1)
    #     save_steps = max(total_steps // num_saving_steps, 1)
    #     logging_steps = max(total_steps // 5, 1)

    #     training_args = TrainingArguments(
    #         run_name=self.run_name,
    #         output_dir=f'{self.experiment_folder}/train/',
    #         **self.training_config.trainer,
    #         evaluation_strategy="steps",
    #         eval_steps=eval_steps,
    #         save_steps=save_steps,
    #         logging_steps=logging_steps,
    #         load_best_model_at_end=True,
    #         remove_unused_columns=False,
    #     )


    #     trainer = Trainer(
    #         model=self.generator.model,
    #         args=training_args,
    #         data_collator=self.generator.model.collate_fn,
    #         train_dataset=train_test_datasets['train'],
    #         eval_dataset=train_test_datasets['test']
    #     )

    #     trainer.train()
    #     self.generator.model = trainer.model
    #     move_finished_experiment(self.experiment_folder)
    #     self.experiment_folder = get_finished_experiment_name(self.experiment_folder)
    #     return self.experiment_folder


