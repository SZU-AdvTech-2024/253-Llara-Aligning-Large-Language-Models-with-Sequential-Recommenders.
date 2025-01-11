import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import random
import torch
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer
import os

class BatchProcessor:
    def __init__(self, template_collection=None, tokenizer=None, is_training=False, end_token="\n", total_steps=1):
        self.template_collection = template_collection
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.end_token = end_token
        self.total_steps = total_steps
        self.current_step = 1

    def __call__(self, samples):
        if isinstance(self.template_collection, list):
            selected_template = random.choice(self.template_collection)
            input_prompts = [selected_template] * len(samples) if not isinstance(selected_template, list) else selected_template
        else:
            template = sample.get("instruction_input", None)
            input_prompts = [template] * len(samples) if not isinstance(template, list) else template
        
        progress_ratio = self.current_step/self.total_steps
        use_embeddings = random.random() >= progress_ratio or not self.is_training
        
        processed_prompts = []
        for idx, sample in enumerate(samples):
            current_prompt = input_prompts[idx]
            if '[HistoryHere]' in current_prompt:
                history_items = sample['seq_name']
                history_marker = ' [HistoryEmb]' if use_embeddings else ' [PH]'
                history_text = ", ".join([item + history_marker for item in history_items])
                current_prompt = current_prompt.replace('[HistoryHere]', history_text)
            
            if '[CansHere]' in current_prompt:
                candidates = sample['cans_name']
                candidate_marker = ' [CansEmb]' if use_embeddings else ' [PH]'
                candidates_text = ", ".join([item + candidate_marker for item in candidates])
                current_prompt = current_prompt.replace('[CansHere]', candidates_text)
            
            processed_prompts.append(current_prompt)
        
        self.current_step += 1
        answer_texts = [sample['correct_answer'] for sample in samples]

        if self.is_training:
            terminated_answers = [text + self.end_token for text in answer_texts]
            prompt_pairs = [[p, t] for p, t in zip(processed_prompts, terminated_answers)]
            encoded_batch = self.tokenizer(
                prompt_pairs,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True
            )
            
            return {
                "tokens": encoded_batch,
                "seq": torch.stack([torch.tensor(s['seq']) for s in samples]),
                "cans": torch.stack([torch.tensor(s['cans']) for s in samples]),
                "len_seq": torch.stack([torch.tensor(s['len_seq']) for s in samples]),
                "len_cans": torch.stack([torch.tensor(s['len_cans']) for s in samples]),
                "item_id": torch.stack([torch.tensor(s['item_id']) for s in samples]),
                "flag": not use_embeddings
            }
        
        encoded_batch = self.tokenizer(
            processed_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=True
        )
        
        return {
            "tokens": encoded_batch,
            "seq": torch.stack([torch.tensor(s['seq']) for s in samples]),
            "cans": torch.stack([torch.tensor(s['cans']) for s in samples]),
            "len_seq": torch.stack([torch.tensor(s['len_seq']) for s in samples]),
            "len_cans": torch.stack([torch.tensor(s['len_cans']) for s in samples]),
            "item_id": torch.stack([torch.tensor(s['item_id']) for s in samples]),
            "correct_answer": answer_texts,
            "cans_name": [s['cans_name'] for s in samples]
        }

class DataInterface(pl.LightningDataModule):
    def __init__(self, tokenizer=None, num_workers=8, dataset='', **kwargs):
        super().__init__()
        self.workers = num_workers
        self.tokenizer = tokenizer
        self.dataset_name = dataset
        self.config = kwargs
        self.batch_size = kwargs['batch_size']
        self.epochs = kwargs['max_epochs']
        self._initialize_components()
        self._load_templates(kwargs['prompt_path'])
        
        self.train_dataset = self._create_dataset('train')
        self.val_dataset = self._create_dataset('val')
        self.test_dataset = self._create_dataset('test')
        self.total_steps = self.epochs * (len(self.train_dataset)//self.batch_size) // self.workers

    def _initialize_components(self):
        module_name = self.dataset_name
        class_name = ''.join([part.capitalize() for part in module_name.split('_')])
        try:
            self.dataset_class = getattr(importlib.import_module(
                '.'+module_name, package=__package__), class_name)
        except:
            raise ValueError(f'Failed to load dataset: data.{module_name}.{class_name}')

    def _create_dataset(self, stage, **extra_args):
        constructor_params = inspect.getargspec(self.dataset_class.__init__).args[1:]
        init_args = {k: v for k, v in self.config.items() if k in constructor_params}
        init_args.update(extra_args)
        init_args['stage'] = stage
        return self.dataset_class(**init_args)

    def _load_templates(self, template_path):
        if os.path.isfile(template_path):
            with open(template_path, 'r') as file:
                self.templates = [line.strip() for line in file.readlines()]
            print(f'Loaded {len(self.templates)} templates')
            print(f'Example template:\n{random.choice(self.templates)}')
        else:
            self.templates = []

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=True,
            drop_last=True,
            collate_fn=BatchProcessor(
                template_collection=self.templates,
                tokenizer=self.tokenizer,
                is_training=True,
                total_steps=self.total_steps
            )
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=False,
            collate_fn=BatchProcessor(
                template_collection=self.templates,
                tokenizer=self.tokenizer,
                is_training=False
            )
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            shuffle=False,
            collate_fn=BatchProcessor(
                template_collection=self.templates,
                tokenizer=self.tokenizer,
                is_training=False
            )
        )