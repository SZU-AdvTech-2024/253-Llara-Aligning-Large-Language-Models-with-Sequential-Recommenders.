import inspect
import torch
import importlib
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl

from transformers import LlamaForCausalLM, LlamaTokenizer
import random
from pandas.core.frame import DataFrame
import os.path as op
import os
from optims import LinearWarmupCosineLRScheduler
import numpy as np
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel

class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._initialize_models()
        
    def _initialize_models(self):
        self._init_llm()
        self._init_rec_model()
        self._init_projector()
        
    def _init_llm(self):
        self.llama_tokenizer = self._setup_tokenizer()
        self.llama_model = self._setup_llm_model()
        self._configure_llm_tuning()
        
    def _setup_tokenizer(self):
        tokenizer = LlamaTokenizer.from_pretrained(self.hparams.llm_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({
            'pad_token': '[PAD]',
            'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']
        })
        tokenizer.padding_side = "right"
        return tokenizer
    
    def _setup_llm_model(self):
        model = LlamaForCausalLM.from_pretrained(self.hparams.llm_path, torch_dtype=torch.bfloat16)
        model.resize_token_embeddings(len(self.llama_tokenizer))
        return model
    
    def _configure_llm_tuning(self):
        tuning_strategies = {
            'lora': self._apply_lora_tuning,
            'freeze': self._apply_freeze_tuning,
            'freeze_lora': self._apply_freeze_lora_tuning
        }
        if self.hparams.llm_tuning not in tuning_strategies:
            raise NotImplementedError(f"Unsupported tuning strategy: {self.hparams.llm_tuning}")
        tuning_strategies[self.hparams.llm_tuning]()
        
    def _apply_lora_tuning(self):
        if self.hparams.peft_dir:
            self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
        else:
            peft_config = self._get_peft_config()
            self.peft_config = peft_config
            self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()
        
    def _get_peft_config(self):
        if self.hparams.peft_config:
            return LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            lora_dropout=self.hparams.lora_dropout,
            target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        )

    def forward(self, batch):
        targets = batch["tokens"].input_ids.masked_fill(
            batch["tokens"].input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:,1:], -100)
        input_embeds = self._process_embeddings(batch)
        return self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False
        )

    def generate(self, batch, temperature=0.8, do_sample=False, num_beams=1, max_gen_length=64, 
                min_gen_length=1, repetition_penalty=1.0, length_penalty=1.0, num_return_sequences=1):
        input_embeds = self._process_embeddings(batch)
        generate_ids = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )
        return [text.strip() for text in self.llama_tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)]

    def _process_embeddings(self, batch):
        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"].input_ids)
        special_tokens = self._get_special_token_mappings()
        
        for i in range(input_embeds.size(0)):
            for token_id, (embeddings_func, length_key) in special_tokens.items():
                indices = (batch["tokens"].input_ids[i] == token_id).nonzero().view(-1)
                if indices.shape[0] > 0:
                    embeddings = embeddings_func(batch)
                    if length_key:
                        length = batch[length_key][i].item()
                        for idx, emb in zip(indices[:length], embeddings[i,:length]):
                            input_embeds[i,idx] = emb
                    else:
                        input_embeds[i,indices[0]] = embeddings[i]
        return input_embeds

    def _get_special_token_mappings(self):
        return {
            self.llama_tokenizer("[HistoryEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item():
                (lambda batch: self.encode_items(batch["seq"]), "len_seq"),
            self.llama_tokenizer("[CansEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item():
                (lambda batch: self.encode_items(batch["cans"]), "len_cans"),
            self.llama_tokenizer("[ItemEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item():
                (lambda batch: self.encode_items(batch["item_id"]), None)
        }

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        self._update_projector_grad_state(batch["flag"])
        loss = self(batch).loss
        self._log_training_metrics(loss)
        return loss

    def _update_projector_grad_state(self, flag):
        for param in self.projector.parameters():
            param.requires_grad = not flag

    def _log_training_metrics(self, loss):
        self.log('loss', loss)
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('global_step_num', self.trainer.global_step, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.projector.parameters(), 'lr': self.hparams.lr, 
             'weight_decay': getattr(self.hparams, 'weight_decay', 0)},
            {'params': self.llama_model.parameters(), 'lr': self.hparams.lr}
        ])

        if not self.hparams.lr_scheduler:
            return optimizer

        if self.hparams.lr_scheduler == 'cosine':
            max_steps = self.trainer.max_steps
            warmup_steps = max_steps // 20
            self.scheduler = LinearWarmupCosineLRScheduler(
                optimizer,
                max_step=max_steps,
                min_lr=self.hparams.lr_decay_min_lr,
                init_lr=self.hparams.lr,
                warmup_steps=warmup_steps,
                warmup_start_lr=self.hparams.lr_warmup_start_lr
            )
            return optimizer
        raise ValueError('Invalid lr_scheduler type!')

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.save == 'part':
            checkpoint.pop('optimizer_states')
            checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items() 
                                      if hasattr(self, k.split('.')[0]) and 
                                      getattr(self, k.split('.')[0]).requires_grad}

    def on_validation_epoch_start(self):
        self.val_content={
            "generate":[],
            "real":[],
            "cans":[],
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        generate_output = self.generate(batch)
        output=[]
        for i,generate in enumerate(generate_output):
            real=batch['correct_answer'][i]
            cans=batch['cans_name'][i]
            generate=generate.strip().split("\n")[0]
            output.append((generate,real,cans))
        return output

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate,real,cans in outputs:
            self.val_content["generate"].append(generate)
            self.val_content["real"].append(real)
            self.val_content["cans"].append(cans)

    def on_validation_epoch_end(self):
        df=DataFrame(self.val_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'valid.csv'))
        prediction_valid_ratio,hr=self.calculate_hr1(self.val_content)
        metric=hr*prediction_valid_ratio
        self.log('val_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_content={
            "generate":[],
            "real":[],
            "cans":[],
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        generate_output = self.generate(batch)
        output=[]
        for i,generate in enumerate(generate_output):
            real=batch['correct_answer'][i]
            cans=batch['cans_name'][i]
            generate=generate.strip().split("\n")[0]
            output.append((generate,real,cans))
        return output
    
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate,real,cans in outputs:
            self.test_content["generate"].append(generate)
            self.test_content["real"].append(real)
            self.test_content["cans"].append(cans)

    def on_test_epoch_end(self):
        df=DataFrame(self.test_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'test.csv'))
        prediction_valid_ratio,hr=self.calculate_hr1(self.test_content)
        metric=hr*prediction_valid_ratio
        self.log('test_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)

    def load_projector(self):
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.projector = self.instancialize(Model, rec_size=self.hparams.rec_size, llm_size=self.llama_model.config.hidden_size)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

    def load_rec_model(self, rec_model_path):
        print('Loading Rec Model')
        self.rec_model = torch.load(rec_model_path, map_location="cpu")
        self.rec_model.eval()
        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False
        print('Loding Rec model Done')

    def encode_items(self, seq):
        if self.hparams.rec_embed=="SASRec":
            item_rec_embs=self.rec_model.cacu_x(seq)
        elif self.hparams.rec_embed in ['Caser','GRU']:
            item_rec_embs=self.rec_model.item_embeddings(seq)
        item_txt_embs=self.projector(item_rec_embs)
        return item_txt_embs
    
    def embed_tokens(self, token_ids):
        embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    def wrap_emb(self, batch):
        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"].input_ids)
        
        his_token_id=self.llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        cans_token_id=self.llama_tokenizer("[CansEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        item_token_id=self.llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        his_item_embeds= self.(batch["seq"])
        cans_item_embeds= self.encode_items(batch["cans"])
        item_embeds=self.encode_items(batch["item_id"])
            
        for i in range(len(batch["len_seq"])):
            if (batch["tokens"].input_ids[i]==his_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"].input_ids[i]==his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:batch["len_seq"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["tokens"].input_ids[i]==cans_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"].input_ids[i]==cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,cans_item_embeds[i,:batch["len_cans"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["tokens"].input_ids[i]==item_token_id).nonzero().shape[0]>0:
                idx=(batch["tokens"].input_ids[i]==item_token_id).nonzero().item()
                input_embeds[i,idx]=item_embeds[i]
        return input_embeds
     
    def calculate_hr1(self,eval_content):
        correct_num=0
        valid_num=0
        total_num=0
        for i,generate in enumerate(eval_content["generate"]):
            real=eval_content["real"][i]
            cans=eval_content["cans"][i]
            total_num+=1
            generate=generate.strip().lower().strip()
            real=real.strip().lower().strip()
            cans=[item.strip().lower().strip() for item in cans]
            gen_cans_list=[]
            for cans_item in cans:
                if cans_item in generate:
                    gen_cans_list.append(cans_item)
            if len(gen_cans_list)==1:
                valid_num+=1
                if real == gen_cans_list[0]:
                    correct_num+=1
        valid_ratio=valid_num/total_num
        if valid_num>0:
            hr1=correct_num/valid_num
        else:
            hr1=0
        return valid_ratio,hr1