from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PretrainedConfig, AutoModel
import torch
import math 
from peft import get_peft_model, LoraConfig, TaskType
import os

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


class BERT_Compressor(torch.nn.Module):
    def __init__(self, compr_model_name, compr_rate, compr_linear_type, decoder_hidden_size):
        super().__init__()
        # init model
        self.model_name = compr_model_name # base model name of BERT; example: bert-base-ucased
        self.model = AutoModel.from_pretrained(compr_model_name, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(compr_model_name, use_fast=True) 
        self.compr_rate = compr_rate # compression rate
        self.compressing_mode = compr_linear_type # linear layer type, could be either concat or mean.

        if self.compressing_mode == 'concat': # default setting in paper
            self.linear = torch.nn.Linear(self.model.config.hidden_size*self.compr_rate, decoder_hidden_size) 
        elif self.compressing_mode == 'mean':
            self.linear = torch.nn.Linear(self.model.config.hidden_size, decoder_hidden_size)
        self.linear = self.linear.bfloat16()

    def forward(self, input_ids, attention_mask):
        # compressing context using BERT
        segment_compress_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True) 
        num_embs = math.ceil(input_ids.size(1) / self.compr_rate)
        all_hidden_states_emb = list()
        if self.compressing_mode == 'concat':
            for segment_idx in range(num_embs):
                start_idx = segment_idx * self.compr_rate
                end_idx = (segment_idx + 1) * self.compr_rate
                hidden_state = segment_compress_outputs.hidden_states[-1][:, start_idx:end_idx, :]
                hidden_state_concat = torch.flatten(hidden_state, start_dim=1) #batch_size, hidden_state_dim * compression_rate
                all_hidden_states_emb.append(hidden_state_concat)
        elif self.compressing_mode == "mean":
            for segment_idx in range(num_embs):
                start_idx = segment_idx * self.compr_rate
                end_idx = (segment_idx + 1) * self.compr_rate
                hidden_state = segment_compress_outputs.hidden_states[-1][:, start_idx:end_idx, :]
                # Apply mean pooling to get the final embedding for the segment
                all_hidden_states_emb.append(hidden_state)
        else: 
            raise NotImplementedError()
        
        all_hidden_states_emb_cat = torch.stack(all_hidden_states_emb, dim=1)
        transformed_embeds = self.linear(all_hidden_states_emb_cat)
        

        if self.compressing_mode == "mean":
            transformed_embeds = torch.mean(transformed_embeds, dim=2)

        # dimention of transformed_embeds: (batch_size*generation_top_k, num_embs, decoder_hidden_size)
        return  transformed_embeds

class COCOMConfig(PretrainedConfig):

    model_type = "COCOM"
    def __init__(self,
                decoder_model_name="meta-llama/Llama-2-7b-chat-hf",
                quantization = 'no', 
                generation_top_k = 1, 
                sep = False,
                compr_model_name = "bert-base-uncased", 
                compr_rate = 64,
                compr_linear_type = 'concat',
                lora = False,
                training_form="both",
                lora_r=16,
                 **kwargs):
        super().__init__(**kwargs)

        self.decoder_model_name = decoder_model_name # model name of decoder
        self.quantization = quantization # quantization, could be no, int4, int8
        self.generation_top_k = generation_top_k # top k for each query, for pretraining, set to 1
        self.sep = sep # boolean type, whether to use sep token
        self.compr_model_name = compr_model_name # model name of compressor
        self.compr_rate = compr_rate # compression rate
        self.compr_linear_type = compr_linear_type # linear layer type, could be either concat or mean
        self.lora = lora # boolean type, whether to use lora trsining
        self.training_form = training_form # training form, could be compressor: training only comprssor; both: 
        self.lora_r = lora_r # lora_r for lora training, we use 16 throughout the experiment.

class COCOM(PreTrainedModel):
    config_class = COCOMConfig
    def __init__(self, cfg):
        super().__init__(cfg)
        # define models
        # model could be loaded in three quantization modes: no, int4, int8
        if cfg.quantization == "no":
            self.decoder = AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2", 
                low_cpu_mem_usage = True,
                )
        elif cfg.quantization == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16',
                low_cpu_mem_usage = True,
            )
            self.decoder = AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name, 
                quantization_config=quant_config,
                attn_implementation="flash_attention_2", 
                torch_dtype=torch.bfloat16,
                resume_download=True,
                low_cpu_mem_usage = True,
                trust_remote_code=True,
            )
        elif cfg.quantization == "int8":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                bnb_4bit_compute_dtype='bfloat16',
                low_cpu_mem_usage = True,
            )
            self.decoder = AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                quantization_config=quant_config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                resume_download=True,
                low_cpu_mem_usage = True,
                trust_remote_code=True,
            )
        else:
            raise NotImplementedError()
        
        # when compr_model_name is not set, then means using a decoder-based compressor, otherwise a bert based compressor
        if cfg.compr_model_name is not None:
            # case bert based compressor
            self.compr = BERT_Compressor(cfg.compr_model_name, cfg.compr_rate, cfg.compr_linear_type, self.decoder.config.hidden_size)
        else:
            # case decoder based compressor
            self.compr = None

        # set lora adaptors
        if cfg.lora:
            peft_config = LoraConfig(
                        task_type="CAUSAL_LM",
                        r=cfg.lora_r,
                        lora_alpha=2* cfg.lora_r,
                        target_modules='all-linear',
                        lora_dropout=0.1,
                    )
            self.decoder = get_peft_model(self.decoder, peft_config)
            self.decoder.print_trainable_parameters()  

        # for training_form=compressor, then freeze the decoder for BERT-based
        self.training_form = cfg.training_form
        if self.training_form == "compressor" and self.compr is not None:
            freeze_model(self.decoder)

        self.decoder_tokenizer = AutoTokenizer.from_pretrained(cfg.decoder_model_name, use_fast=True, padding_side='left')

        # define special tokens
        self.decoder_tokenizer.add_special_tokens({'additional_special_tokens': ['<MEM>', '<AE>', '<ENC>', '<SEP>']})
        self.decoder_tokenizer.mem_token = '<MEM>' # Memory token
        self.decoder_tokenizer.ae_token = '<AE>' # token for autoencoding on decoder side
        self.decoder_tokenizer.enc_token = '<ENC>' # token for autoencoding on compressor side
        self.decoder_tokenizer.sep_token = '<SEP>' # sep token between document

        self.decoder_tokenizer.mem_token_id = self.decoder_tokenizer.convert_tokens_to_ids('<MEM>')
        self.decoder_tokenizer.ae_token_id = self.decoder_tokenizer.convert_tokens_to_ids('<AE>')
        self.decoder_tokenizer.sep_token_id = self.decoder_tokenizer.convert_tokens_to_ids('<SEP>')
        # if pad token ecist then use pad token, othrwise bos token
        if self.decoder_tokenizer.pad_token_id is None:
            self.decoder_tokenizer.pad_token_id = self.decoder_tokenizer.bos_token_id

        # resize the tokenizer embedding
        self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))
        self.decoder.generation_config.top_p=None
        self.decoder.generation_config.temperature=None
        
        self.compr_model_name = cfg.compr_model_name
        # other settings
        self.generation_top_k = cfg.generation_top_k
        self.sep = cfg.sep
        self.compr_rate = cfg.compr_rate
        self.local_rank = os.getenv('LOCAL_RANK', '0')

    def compress_and_replace_emb(self, enc_input_ids, enc_attention_mask, dec_input_ids):
        indices = range(0, enc_input_ids.size(0) + 1, self.generation_top_k)
        if self.compr:
            compressed_embs = self.compr(enc_input_ids, enc_attention_mask)
            input_embeds = self.replace_embeddings(compressed_embs, dec_input_ids, indices)
        else:
            compressed_embs = self.compr_decoder(enc_input_ids, enc_attention_mask)
            input_embeds = self.replace_embeddings(compressed_embs, dec_input_ids, indices)
        return input_embeds
    
    def compr_decoder(self, input_ids, attention_mask):
        emb = self.decoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        mask = input_ids == self.decoder_tokenizer.mem_token_id
        return emb[mask].reshape(emb.size(0), -1, emb.size(-1))
    

    def replace_embeddings(self, compressed_embs, dec_input_ids, indices):
        # Embed the decoder input
        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
        num_embs = compressed_embs.size(1)
        if self.sep:
            slot_len = num_embs + 1
        else:
            slot_len = num_embs
        # get first mem_token inidices
        first_mem_token_indices = torch.argmax((dec_input_ids == self.decoder_tokenizer.mem_token_id).int(), dim=1)
        batch_size = inputs_embeds.size(0)
        # for each example in batch, replace them with compressed embeddings

        for i in range(batch_size):
            for j in range(indices[i], indices[i + 1]):
                start_idx = first_mem_token_indices[i].item() + (j-indices[i]) * slot_len
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = compressed_embs[j]
        return inputs_embeds


    def forward(self, 
            enc_input_ids: torch.LongTensor = None,
            enc_attention_mask: torch.LongTensor = None,
            dec_input_ids: torch.LongTensor = None, 
            dec_attention_mask: torch.LongTensor = None,
            labels: torch.LongTensor = None):
        # enc_input_ids: stores the contexts, should be flattened from all queries before input, dimention (batch_size*generation_top_k, token_length)
        # enc_attention_mask: attention mask of enc_input_ids
        # dec_input_ids: stores the prompts (including mem tokens), dimention (batch_size, token_length)
        # dec_attention_mask: attention mask of dec_input_ids

        # Perform compression with gradient tracking
        inputs_embeds = self.compress_and_replace_emb(enc_input_ids, enc_attention_mask, dec_input_ids)

        # if training_form is compressor, then detach the inputs_embeds, to make gradient not count in decoder
        if (self.training_form == "compressor") and (self.compr is None):
            inputs_embeds  = inputs_embeds.detach()

        # decoding
        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=dec_attention_mask, labels=labels)

        return {"loss": decoder_outputs.loss, "logits": decoder_outputs.logits}


        
    def generate(self, model_input, max_new_tokens=128):

        enc_input_ids, enc_attention_mask, dec_input_ids, dec_attention_mask = model_input['enc_input_ids'], model_input['enc_attention_mask'], model_input['dec_input_ids'], model_input['dec_attention_mask']
        inputs_embeds = self.compress_and_replace_emb(enc_input_ids.to('cuda'), enc_attention_mask.to('cuda'), dec_input_ids.to('cuda'))


        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds.to("cuda"), 
            attention_mask=dec_attention_mask.to("cuda"),
            do_sample=False,
            top_p=None,
            max_new_tokens=max_new_tokens
            )

        decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return decoded
        
    def generate_from_text(self, contexts, questions, max_new_tokens=128):
       
        # first make sure that every list in contexts are having the same length
        assert len(contexts) == len(questions)
        assert all([len(context) == len(contexts[0]) for context in contexts])

        # prepare inp_enc for compression
        # first flatten the contexts
        self.generation_top_k = len(contexts[0])
        flat_contexts = sum(contexts, [])
        #tokenize the contexts, depending if compr exist or not
        if self.compr is not None:
            inp_enc = self.compr.tokenizer(flat_contexts, padding=True, truncation=True, return_tensors='pt', pad_to_multiple_of=self.compr_rate)
            num_mem_tokens = math.ceil(enc_input['input_ids'].size(1) / self.compr_rate)
        else:
            # first need to add special token in flat_contexts
            flat_contexts = [self.decoder_tokenizer.enc_token + self.decoder_tokenizer.bos_token +  context  + self.decoder_tokenizer.bos_token  for context in flat_contexts]
            inp_enc  = self.decoder_tokenizer(flat_contexts, truncation=True, return_tensors='pt', padding="longest")
            num_mem_tokens = math.ceil(enc_input['input_ids'].size(1)-3 / self.compr_rate)
            mem_tokens = torch.full((enc_input['input_ids'].size(0), num_mem_tokens), self.decoder_tokenizer.mem_token_id, dtype=torch.long)
            inp_enc['input_ids'] = torch.cat([mem_tokens, enc_input['input_ids']], dim=1)
            inp_enc['attention_mask'] = torch.cat([torch.ones_like(mem_tokens), enc_input['attention_mask']], dim=1)
        
        
        # prepare inp_dec
        mem_tokens = self.decoder_tokenizer.mem_token * num_mem_tokens
        if self.sep:
            mem_tokens += self.decoder_tokenizer.sep_token
        
        instr = [self.decoder_tokenizer.bos_token + mem_tokens* self.generation_top_k + '[INST]' + question + '\n[/INST]\n' for question in questions]
        inp_dec = self.decoder_tokenizer(instr, truncation=True, return_tensors='pt', padding="longest")

        # generate
        model_input = {
            'enc_input_ids': inp_enc['input_ids'],
            'enc_attention_mask': inp_enc['attention_mask'],
            'dec_input_ids': inp_dec['input_ids'],
            'dec_attention_mask': inp_dec['attention_mask']
        }

        return self.generate(model_input, max_new_tokens)