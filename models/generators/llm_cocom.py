import torch
import math
import random
from models.generators.generator import Generator
from models.generators.cocom import COCOM, COCOMConfig
from utils import prepare_labels

random.seed(42)
def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

class COCOMLLM(Generator):
    def __init__(self, 
                model_name = "cocom", 
                decoder_model_name="meta-llama/Llama-2-7b-chat-hf",
                max_new_tokens = 128, 
                quantization = 'no', 
                generation_top_k = 1, 
                sep = True,
                compr_model_name = "bert-base-uncased", 
                compr_rate = 64,
                compr_linear_type = 'concat',
                lora = False,
                query_dependant = False,
                training_form="both",
                context_max_length=512,
                max_length=1280,
                test_mode="ft", # either ft or ae, ae is for autoencoding, ft is for fine-tune tasks
                **kwargs,
    ):
        """
            Initializes the COCOMLLM class.
        
            Args:
                model_name (str): Name of the model to be used. If 'cocom', a new model is created.
                decoder_model_name (str): Name of the decoder model.
                max_new_tokens (int): Maximum number of new tokens to be generated.
                quantization (str): Quantization type.
                generation_top_k (int): Top k tokens to consider during generation.
                sep (bool): Whether to use a separator token.
                compr_model_name (str): Name of the compression model.
                compr_rate (int): Compression rate.
                compr_linear_type (str): Type of linear compression.
                lora (bool): Whether to use LoRA.
                query_dependant (bool): Whether the compression is query dependant.
                training_form (str): Specifies the part of the model to train ('decoder', 'compressor', 'linear', 'both').
                context_max_length (int): Maximum length of the context.
                max_length (int): Maximum length of the input.
                test_mode (str): Test mode, either 'ft' for fine-tuning or 'ae' for autoencoding.
                **kwargs: Additional keyword arguments.
        """
        # load a new model if model_name is cocom, othwewise from a pretriained checkpoint
        if model_name == 'cocom':
            cfg = COCOMConfig(
                decoder_model_name=decoder_model_name,
                max_new_tokens=max_new_tokens,
                quantization=quantization,
                generation_top_k=generation_top_k,
                sep=sep,
                compr_model_name=compr_model_name,
                compr_rate=compr_rate,
                compr_linear_type=compr_linear_type,
                lora = lora
                )
            self.model = COCOM(cfg)
        else:

            self.model = COCOM.from_pretrained(model_name, ignore_mismatched_sizes=True)
            self.model.sep = sep
            self.model.generation_top_k = generation_top_k
            self.model.max_new_tokens = max_new_tokens
        
        self.training_form = training_form
        assert self.training_form in ['decoder', 'compressor', 'linear', 'both']
        self.test_mode = test_mode
        assert self.test_mode in ['ft', 'ae']
        if self.test_mode == 'ae':
            self.model.sep = False

        if self.model.compr is not None:
            if self.training_form == 'compressor':
                freeze_model(self.model.decoder)
        self.model_name = model_name
        self.query_dependant = query_dependant
        self.max_new_tokens = max_new_tokens
        self.context_max_length = context_max_length
        self.model_max_length = max_length 
        self.response_token_ids = self.get_response_template_ids()
        print("Response token ids")
        print(self.response_token_ids)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, instr_tokenized):
        # generate based on input from collate function
        return self.model.generate(instr_tokenized, max_new_tokens=self.max_new_tokens)
    
    def get_response(self):
        return '\n[/INST]\n'

    def get_response_template_ids(self):
        response_template = self.get_response()
        return self.model.decoder_tokenizer.encode(response_template, add_special_tokens=False)
        

    def prediction_step(self, model, model_input, label_ids=None):
        # used for training
        output = model.forward(**model_input, labels=label_ids)
        return output['logits'], output['loss']
    
    def collate_fn(self, examples,  eval=False, **kwargs):
            """
            Collates a batch of examples.
            
            Args:
                examples (list): batch from dataset
                eval (bool): Whether the function is being called for evaluation.
                **kwargs: Additional keyword arguments.
            
            Returns:
                dict: Collated batch of data.
            """
            ignore_index = -100
            q_ids = [e['q_id'] for e in examples]
            query = [e['query'] for e in examples]
  
            ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)
            docs = sum([example['doc'] for example in examples], []) # flatten all the docs for encoder input
 
            # first to prepare encoder input
            if self.model.compr is not None:
                # case bert-compressor
                if self.query_dependant:
                    #repeate query by generation_top_k times
                    query_combined = [q for q in query for _ in range(self.model.generation_top_k)]
                    docs = [q + "[SEP]" + d for q, d in zip(query_combined, docs)]
                inp_enc = self.model.compr.tokenizer(docs, return_tensors='pt', padding=True, truncation=True, pad_to_multiple_of=self.model.compr_rate, max_length=self.context_max_length)
                num_mem_tokens = math.ceil(inp_enc['input_ids'].size(1) / self.model.compr_rate)
            else:
                # case decoder-compressor
                # first add bos in the beginning of the input, eos in the end
                if self.query_dependant:
                    #repete query by generation_top_k times
                    query_combined = [q for q in query for _ in range(self.model.generation_top_k)]
                    docs = [q + self.model.decoder_tokenizer.sep_token + d for q, d in zip(query_combined, docs)]
                 
                inp_enc = [self.model.decoder_tokenizer.enc_token + self.model.decoder_tokenizer.bos_token + doc + self.model.decoder_tokenizer.eos_token for doc in docs]
                inp_enc = self.model.decoder_tokenizer(inp_enc, return_tensors='pt', padding="longest", max_length=self.context_max_length+3, truncation=True, add_special_tokens=False)
                num_mem_tokens = math.ceil((inp_enc['input_ids'].size(1)- 3) / self.model.compr_rate)
        
                # append memory tokens to the input
                mem_tokens = torch.full((inp_enc['input_ids'].size(0), num_mem_tokens), self.model.decoder_tokenizer.mem_token_id, dtype=torch.long)
                inp_enc['input_ids'] = torch.cat([inp_enc['input_ids'], mem_tokens], dim=1)
                inp_enc['attention_mask'] = torch.cat([inp_enc['attention_mask'], torch.ones(inp_enc['attention_mask'].size(0), num_mem_tokens)], dim=1)
            
            # input for decoder
            # [Padding][BOS][mem][INST]{question}?[/INST]
            mem_tokens = self.model.decoder_tokenizer.mem_token * num_mem_tokens
            if self.model.sep:
                mem_tokens += self.model.decoder_tokenizer.sep_token

            if eval:
                #for inference

                label = [[e['label']] if isinstance(e['label'], str) else e['label'] for e in examples]
                if self.test_mode == 'ae':
                    instr = [self.model.decoder_tokenizer.ae_token + self.model.decoder_tokenizer.bos_token + mem_tokens * self.model.generation_top_k for q in query]
                else:
                    instr = [self.model.decoder_tokenizer.bos_token + mem_tokens * self.model.generation_top_k + '[INST]' + q + self.get_response() for q in query]
                inp_dec = self.model.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False,
                                        truncation=True,  max_length=self.model_max_length)
            else:
                if self.test_mode == 'ae':
                    # it's not possible to have ae mode for training
                    raise ValueError("AE mode is not possible for training")
                label = [e['label'] if isinstance(e['label'], str) else random.choice(e['label']) for e in examples]
     
                instr = [self.model.decoder_tokenizer.bos_token + mem_tokens * self.model.generation_top_k + '[INST]' + q + self.get_response() + e + self.model.decoder_tokenizer.eos_token  for q, e in zip(query, label)]
                inp_dec = self.model.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True, max_length=self.model_max_length)
                label_ids = prepare_labels(inp_dec["input_ids"], self.response_token_ids[1:], ignore_index=ignore_index)
                # print the ones that has all -100
                for i, label_id in enumerate(label_ids):
                    if all([l == ignore_index for l in label_id]):
                        print(f"Warning: all -100 in label_ids {i}")
                        print(instr[i])
                        print(inp_dec["input_ids"][i])
                        print(label_id)
                    

            data_dict = {}
            if not eval:
                data_dict['label_ids'] =  label_ids

            model_input = {
                'enc_input_ids': inp_enc['input_ids'],
                'enc_attention_mask': inp_enc['attention_mask'],
                'dec_input_ids': inp_dec['input_ids'],
                'dec_attention_mask': inp_dec['attention_mask'],
            }

            data_dict.update({
                'model_input': model_input,
                'q_id': q_ids, 
                'query': query, 
                'instruction': instr,
                'label': label, 
                'ranking_label': ranking_label,
            })
            return data_dict