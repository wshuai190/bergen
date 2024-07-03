import json
import torch
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModelForCausalLM
from peft import LoraConfig
from models.generators.modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments, LoraArguments
import sys
from safetensors.torch import load_file
import random 
from models.generators.generator import Generator

random.seed(42)
device = "cuda"
def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

class LLMICAE(Generator):
    def __init__(self, 
                model_name = "icae", 
                decoder_model_name="mistralai/Mistral-7B-v0.1",
                max_new_tokens = 128, 
                **kwargs
                ):
        model_args = ModelArguments(
            train=False,
            lora_train = True,
            compressor_path= decoder_model_name,
            decoder_path = decoder_model_name,
        )

        training_args = TrainingArguments(output_dir="./")

        lora_config = LoraConfig(
            r=512,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        device = "cuda"

        self.model = ICAE(model_args, training_args, lora_config)
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name

        state_dict = load_file(self.model_name)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
        self.model.decoder_tokenizer.pad_token_id = self.model.decoder_tokenizer.bos_token_id
        self.model.decoder_tokenizer.padding_side = "left"

    def prediction_step(self, model, model_input, label_ids=None):

        prompt_tokenized = [e["input_ids"][0] for e in model_input["prompt"]]
        docs_compression_tokenized = [e["input_ids"][0] if e is not None else None for e in model_input["docs_compression"]]
        label_ids = torch.stack(label_ids)
        prompt_tokenized = torch.stack(prompt_tokenized)
        docs_compression_tokenized = torch.stack(docs_compression_tokenized) if all(e is not None for e in docs_compression_tokenized) else None
        output = model(input_ids=docs_compression_tokenized,prompt_answer_ids=prompt_tokenized ,labels=label_ids)
        return output.logits, output.loss

    def generate(self, instr_tokenized):
        prompts_tokenized = instr_tokenized["prompt"]
        docs = instr_tokenized["docs"]
        batch_size = len(prompts_tokenized)

        # Encoder part

        memory_slots = self.compress_docs(docs)
        #total_max_length = self.calculate_total_max_length(prompts_tokenized, docs_no_compression_tokenized)
        # Decoder part
        prompt_max_length = max([len(prompt) for prompt in prompts_tokenized])

        decoder_input_embeddings_list = [
            self.prepare_decoder_input(prompt, memory_slot, prompt_max_length)
            for prompt, memory_slot in zip(prompts_tokenized, memory_slots)
        ]

        decoder_input_embeddings = torch.cat(decoder_input_embeddings_list, dim=0)

        del decoder_input_embeddings_list
        torch.cuda.empty_cache()
        #get embedding size
        print("Embedding size: ", decoder_input_embeddings.size())
        generated_texts = self.perform_generation(decoder_input_embeddings, batch_size)
        print("Generated texts: ", generated_texts)
        return generated_texts

    def compress_docs(self, docs_tokenized):

        memory_slots = self.model._compress(docs_tokenized.to("cuda"))
        return memory_slots


    def prepare_decoder_input(self, prompt_tokenized, memory_slot, prompt_max_length):
        # Calculate lengths

        padding_needed = prompt_max_length - len(prompt_tokenized)
        
        if padding_needed > 0:
           prompt_left_ids= [self.model.decoder_tokenizer.pad_token_id] * padding_needed + [1, 733, 16289, 28793]
        else:
            prompt_left_ids = [1, 733, 16289, 28793]

        # Initialize prompt_left_ids with specific tokens
        # Convert to tensor
        prompt_left_ids_tensor = torch.LongTensor([prompt_left_ids]).to("cuda")

        # Right padding
        prompt_right_ids = [self.model.ft_token_id] + prompt_tokenized + [733, 28748, 16289,28793]
        prompt_right_ids_tensor = torch.LongTensor([prompt_right_ids]).to("cuda")

        # Get embeddings
        prompt_left_embs = self.model.tokens_to_embeddings(prompt_left_ids_tensor)
        prompt_right_embs = self.model.tokens_to_embeddings(prompt_right_ids_tensor)

        # Concatenate embeddings
        if memory_slot is None:
            decoder_input_embedding = torch.cat((prompt_left_embs, prompt_right_embs), dim=1)
        else:
            memory_slot_tensor = memory_slot.unsqueeze(0).to("cuda")
            decoder_input_embedding = torch.cat((prompt_left_embs, memory_slot_tensor, prompt_right_embs), dim=1)
            del memory_slot_tensor
        del prompt_left_ids_tensor, prompt_right_ids_tensor, prompt_left_embs, prompt_right_embs
        torch.cuda.empty_cache()
        return decoder_input_embedding

    def perform_generation(self, decoder_input_embeddings, batch_size):
        past_key_values = None
        generated_texts = [[] for _ in range(batch_size)]
        output = decoder_input_embeddings

        break_list = [False] * batch_size
        for _ in range(self.max_new_tokens):
            with self.model.icae.disable_adapter():
                out = self.model.icae(inputs_embeds=output.half(), past_key_values=past_key_values, use_cache=True)
                logit = out.logits[:, -1, :self.model.vocab_size - 1]
                past_key_values = out.past_key_values
                next_token_id = torch.argmax(logit, dim=-1)

                # Update break_list and generated_texts
                for j, token_id in enumerate(next_token_id):
                    if token_id.item() == 2:  # Assuming 2 is the eos_token_id
                        break_list[j] = True
                    elif not break_list[j]:
                        generated_texts[j].append(token_id.item())

                # Check if all sequences are complete
                if all(break_list):
                    break

                # Prepare the next input token
                output = self.model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(device)

        # Decode the generated token ids to text
        generated_texts = self.model.decoder_tokenizer.batch_decode(generated_texts, skip_special_tokens=True)
        print(generated_texts)
        return generated_texts

    def collate_fn(self, examples, eval=False, **kwargs):
        ignore_index = -100
        q_ids = [e['q_id'] for e in examples]
        query = [e['query'] for e in examples]
        instr = query
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)
        docs = [" ".join(example['doc']) for example in examples]
        tokenized_prompt = self.model.decoder_tokenizer(query, truncation=True)
        input_ids_prompt = tokenized_prompt['input_ids']
        tokenized_docs = self.model.decoder_tokenizer(docs, truncation=True, padding="longest", return_tensors="pt", max_length=512)
        input_ids_docs = tokenized_docs['input_ids']

        input_ids_list = {"prompt": input_ids_prompt, "docs": input_ids_docs}

        label = [[e['label']] if isinstance(e['label'], str) else e['label'] for e in examples]

        data_dict = {}
        if not eval:
            label_ids = [e["tokenized_input"]["label"] for e in examples]
            data_dict['label_ids'] = label_ids

        data_dict.update({
            'model_input': input_ids_list,
            'q_id': q_ids,
            'query': query,
            'label': label,
            'ranking_label': ranking_label,
            'instruction': instr,
        })

        # Remove other tensors
        del input_ids_prompt, input_ids_docs
        torch.cuda.empty_cache()

        return data_dict