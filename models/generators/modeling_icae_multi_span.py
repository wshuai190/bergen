# ICAE that supports multi span concat
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from peft import (
    get_peft_model,
)
import math
from safetensors.torch import load_file

import numpy as np
from rouge import Rouge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ModelArguments:
    #model_name_or_path: str = field(default="mistralai/Mistral-7B-v0.1")
    compressor_path: str = field(default="mistralai/Mistral-7B-v0.1")
    decoder_path: str = field(default="mistralai/Mistral-7B-v0.1")
    train: bool = field(
        default=True,
        metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"}
    )
    train_mode: str = field(
        default="compressor",
        metadata={"help": "compressor or decoder or both or same"}
    )

    lora_train: bool = field(
        default=False,
        metadata={"help": "if true, the lora model ckpt will be initialized for training; else, it's for inference"}
    )

    model_type: str = field(
        default="ICAE",
        metadata={"help": "Model type, ICAE or ICAE_NEW"}
    )

    compressing_mode: str = field(
        default="cls_embed",
        metadata={"help": "all_embed or cls_embed"}

    )

    sep: bool = field(
        default=False,
        metadata={"help": "if true, the has sep between docs"}
    )
@dataclass
class LoraArguments:
    r: int = field(
        default=128,
        metadata={"help": "lora rank"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "lora dropout"}
    )

    lora_alpha: int = field(
        default=32,
        metadata={"help": "lora alpha"}
    )

    bias: str = field(
        default="none",
        metadata={"help": "lora bias"}
    )

    task_type: str = field(
        default="CAUSAL_LM",
        metadata={"help": "lora task type"}
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    debug_data: bool = field(default=False, metadata={"help": "Enable debug dataset to quickly verify the training process"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default="output")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    fixed_mem_size: int = field(
        default=128,
        metadata={"help": "Enalbing the fixed mem size."},
    )
    mean_compression_rate: int = field(
        default=4,
        metadata={"help": "Mean compression rate; default=32"},
    )
    min_tokens_for_lm: int = field(
        default=64,
        metadata={"help": "Minimum tokens for lm objective learning"},
    )
    leave_tokens_for_lm: int = field(
        default=8,
        metadata={"help": "Leave some tokens without loss for lm objective"},
    )
    lm_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio for LM training."},
    )
    add_special_token_for_lm: bool = field(
        default=False,
        metadata={"help": "Add a special token for the prompt of language modeling; default: False"},
    )
    restore_from: str = field(
        default="",
        metadata={"help": "The checkpoint that should be restored from for fine-tuning"}
    )
    train_from_all_docs: bool = field(
        default=False,
        metadata={"help": "Train from all documents"}
    )

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class ICAE(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.compressor_path = model_args.compressor_path
        self.decoder_path = model_args.decoder_path
        self.train_mode = model_args.train_mode
        self.lora_train = model_args.lora_train

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.icae = AutoModelForCausalLM.from_pretrained(self.compressor_path, torch_dtype=torch.float16 if training_args.bf16 is False else torch.bfloat16, use_flash_attention_2=True, resume_download=True)
        
        self.training = self.model_args.train    
        
        #if self.training:    # indepedent model for gradient checkpointing
        #self.decoder = self.icae
        if self.train_mode == "same":
            self.decoder = self.icae
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(self.decoder_path, torch_dtype=torch.float16 if training_args.bf16 is False else torch.bfloat16, use_flash_attention_2=True, resume_download=True)


        self.vocab_size = self.icae.config.vocab_size + 1    # [PAD] token
        self.encoder_pad_token_id = self.vocab_size - 1
        self.decoder_pad_token_id = self.vocab_size - 1
        self.mean_compression_rate = training_args.mean_compression_rate

        # tunable
        self.mem_size = self.training_args.fixed_mem_size
        self.vocab_size_with_mem = self.vocab_size + self.mem_size # so, the mem tokens are in the range [self.vocab_size, self.vocab_size + self.mem_size)

        # special tokens in addition to mem and length tokens
        self.ae_token_id = self.vocab_size_with_mem + 0
        self.lm_token_id = self.vocab_size_with_mem + 1
        self.ft_token_id = self.vocab_size_with_mem + 2        

        self.icae.resize_token_embeddings(self.vocab_size_with_mem + 3) 
        
        # special tokens for Llama-2/Mistral tokenizer
        self.bos_id = 1
        self.eos_id = 2
        
        self.dim = self.icae.config.hidden_size
        if self.lora_train:
            if self.train_mode == "compressor":
                self.icae = get_peft_model(self.icae, lora_config)
            elif self.train_mode == "decoder":
                self.decoder = get_peft_model(self.decoder, lora_config)
            elif self.train_mode == "both":
                self.icae = get_peft_model(self.icae, lora_config)
                self.decoder = get_peft_model(self.decoder, lora_config)
            elif self.train_mode == "same":
                self.icae = get_peft_model(self.icae, lora_config)
                self.decoder = self.icae

        #self.icae = get_peft_model(self.icae, lora_config)


        self.memory_token_embed = nn.Embedding(self.mem_size + 3, self.dim, padding_idx=None)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        self.icae_tokenizer = AutoTokenizer.from_pretrained(self.compressor_path, use_fast=False)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(self.decoder_path, use_fast=False) if self.train_mode != "same" else self.icae_tokenizer

        self.append_sequence = torch.arange(self.vocab_size, self.vocab_size + self.mem_size, dtype=torch.long, device=device).unsqueeze(0)    # mem tokens

        if self.training:
            self.init()


    def init(self):
        if self.train_mode == "compressor":
            print("Freezing decoder...")
            freeze_model(self.decoder)
        elif self.train_mode == "decoder":
            print("Freezing compressor...")
            freeze_model(self.icae)

        if self.training_args.restore_from is not None and self.training_args.restore_from != "":
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")
        print("Enabling gradient checkpointing...")
        # self.icae.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                
        
    def compute_num_segments(self, total_length):
        assert total_length > 0
        num_segments = math.ceil(total_length / (self.mem_size * self.mean_compression_rate))
        return num_segments


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        prompt_answer_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # encoder part
        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)
        prompt_answer_embs = self.decoder.get_input_embeddings()(prompt_answer_ids)
        #prompt_answer_embs = self.icae.get_base_model().model.embed_tokens(prompt_answer_ids)
        max_compressed_length = num_segments * self.mem_size
        compress_outputs = torch.zeros((batch_size, max_compressed_length, self.dim)).to(prompt_answer_embs)
        actual_append_sequence = self.append_sequence.repeat(batch_size, 1)

        #print("size of compress_outputs" + str(compress_outputs.size()))
        
        for segment_idx in range(num_segments):

            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            segment_input_ids = torch.cat([segment_input_ids, actual_append_sequence.to(segment_input_ids)], dim=1)
            #print("size of segment_input_ids" + str(segment_input_ids.size()))
            mem_flag = segment_input_ids >= self.vocab_size

            #segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)

            segment_input_embedding = self.icae.get_input_embeddings()(segment_input_ids)
            segment_input_embedding[mem_flag] = self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size).to(segment_input_embedding)

            # compress the current segment
            segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True).hidden_states[-1]

            #print("size of memory flag" + str(mem_flag.size()))
            #print("size of segment_compress_outputs" + str(segment_compress_outputs.size()))

            selected_compress_outputs = torch.stack([
                segment_compress_outputs[i, mem_flag[i]] for i in range(batch_size)
            ])
            #print("size of selected_compress_outputs" + str(selected_compress_outputs.size()))
            # collect memory tokens
            compress_outputs[:, segment_idx*self.mem_size: self.mem_size*(segment_idx+1)] = selected_compress_outputs
        del segment_input_ids, segment_input_embedding, segment_compress_outputs, selected_compress_outputs
        torch.cuda.empty_cache()
            
        # decoder part
        #print("size of compress_outputs" + str(compress_outputs.size()))

        #decoder_mem_flag = (prompt_answer_ids >= self.vocab_size) & (prompt_answer_ids < self.vocab_size + self.mem_size)   # only mem tokens
        #print(decoder_mem_flag[0])
        #print("size of prompt_answer_embs" + str(prompt_answer_embs.size()))
        #print("size of decoder_mem_flag" + str(decoder_mem_flag.size()))
        #expanded_decoder_mem_flag = decoder_mem_flag.unsqueeze(-1).expand(-1, -1, self.dim)
        #print("size of expanded_decoder_mem_flag" + str(expanded_decoder_mem_flag.size()))

        prompt_answer_embs[:, :max_compressed_length, : ] = compress_outputs # replace memory slots
        special_prompt = prompt_answer_ids >= self.vocab_size_with_mem
        prompt_answer_embs[special_prompt] = self.memory_token_embed(prompt_answer_ids[special_prompt] - self.vocab_size).to(prompt_answer_embs)    # replace special token's embedding from self.memory_token_embed
        
        #if self.training:   # has an independent se.f.decoder
            #decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, output_hidden_states=True)
        #else:
            #with self.icae.disable_adapter():   # no independent decoder; use self.icae
        decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, output_hidden_states=True)


        logits = decoder_outputs.logits
        effective_logits = logits[:,:-1,:].reshape(-1, logits.size(-1))
        target_ids = labels[:,1:].reshape(-1)
        loss = self.loss_fct(effective_logits, target_ids)

        return {"loss": loss, "logits": logits, "labels": labels}
    
    
    def tokens_to_embeddings(self, token_ids):   # input_tokens can be either normal tokens and special tokens
        #embeddings = self.icae.get_base_model().model.embed_tokens(token_ids)
        embeddings = self.icae.get_input_embeddings()(token_ids)
        special_flags = token_ids >= self.vocab_size
        embeddings[special_flags] = self.memory_token_embed(token_ids[special_flags] - self.vocab_size).to(embeddings)    # replace special token's embedding from self.memory_token_embed
        return embeddings
        
    
    def _compress(
        self,
        input_ids: torch.LongTensor = None
    ):  # for inference; compress a fixed length of input into memory slots

        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)
        
        max_compressed_length = num_segments * self.mem_size
        compress_outputs = torch.zeros((batch_size, max_compressed_length, self.dim)).to(input_ids)
        actual_append_sequence = self.append_sequence.repeat(batch_size, 1)

        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            segment_input_ids = torch.cat([segment_input_ids, actual_append_sequence], dim=1)
            mem_flag = segment_input_ids >= self.vocab_size

            #segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
            segment_input_embedding = self.icae.get_input_embeddings()(segment_input_ids)
            segment_input_embedding[mem_flag] = self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size).to(segment_input_embedding)

            # compress the current segment
            segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True)
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]
            selected_compress_outputs = torch.stack([
                segment_compress_outputs[i, mem_flag[i]] for i in range(batch_size)
            ])
            # collect memory tokens
            compress_outputs[:, segment_idx*self.mem_size: self.mem_size*(segment_idx+1)] = selected_compress_outputs
            
            # del segment_input_ids, segment_input_embedding
            # torch.cuda.empty_cache()
        
        return compress_outputs
    def compute_metrics(self, eval_pred):
        # tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        logits, labels = eval_pred
        if isinstance(logits, tuple):  # Check if logits are wrapped in a tuple
            logits = logits[0]  # Adjust this to access the correct tuple element
        predictions = logits.argmax(-1)  # Now logits should be a tensor

        return {"exact_match": self.bleu_score_metric(predictions, labels)}

    def bleu_score_metric(self, predictions, target_labels):

        #first remove tokens not in the tokenizer's vocabulary so it nees to be >= 0 and < vocab_size
        # Efficiently remove the ignore index and preserve batch structure
        filtered_predictions = []
        filtered_labels = []
        for i, (pred, label) in enumerate(zip(predictions, target_labels)):
            filtered_predictions.append([token for token in pred if token >= 0 and token < self.decoder_tokenizer.vocab_size])
            filtered_labels.append([token for token in label if token >= 0 and token < self.decoder_tokenizer.vocab_size])


        decoded_predictions = [self.decoder_tokenizer.decode(seq, skip_special_tokens=True) for seq in
                               filtered_predictions]
        print("decoded_predictions", decoded_predictions[0])
        decoded_labels = [[self.decoder_tokenizer.decode(seq, skip_special_tokens=True)] for seq in filtered_labels]
        print("decoded_labels", decoded_labels[0])
        # Calculate BLEU score using calculate_rough
        roughes = []
        for j in range(len(decoded_predictions)):
            roughes.append(self.calculate_rough(decoded_predictions[j], decoded_labels[j][0]))
        bleu_score = np.mean(roughes)

        # print("bleu_score_1", bleu_score_1)
        #
        # decoded_predictions = [seq.split() for seq in decoded_predictions]
        # decoded_labels = [[seq[0].split()] for seq in decoded_labels]
        # except:
        #     #get the maximum of id and the minimum of id token
        #     predictions_flat = [token for seq in filtered_predictions for token in seq]
        #
        #     labels_flat = [token for seq in filtered_labels for token in seq]
        #     max_id = max(max(predictions_flat), max(labels_flat))
        #     min_id = min(min(predictions_flat), min(labels_flat))
        #     print(max_id, min_id)
        #     raise ValueError("The model has generated tokens that are not in the tokenizer's vocabulary. ")


        # Calculate BLEU score using NLTK
        # smoothie = SmoothingFunction().method4
        # bleu_score = corpus_bleu(decoded_labels, decoded_predictions,
        #                          smoothing_function=smoothie) * 100  # Multiply by 100 for percentage

        print("bleu_score", bleu_score)

        return bleu_score

    def calculate_rough(self, prediction, target):

        rouge = Rouge()
        rouge_score = rouge.get_scores(prediction, target, avg=True)
        rouge_score = rouge_score["rouge-l"]["f"]
        return rouge_score