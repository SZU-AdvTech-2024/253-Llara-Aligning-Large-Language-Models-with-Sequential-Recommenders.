import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from model.model_interface import MInterface
from data.data_interface import DInterface
from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU
from SASRecModules_ori import *
from transformers import LlamaForCausalLM, LlamaTokenizer

def setup_callbacks(config):
    callback_list = []
    
    early_stop = plc.EarlyStopping(
        monitor='metric',
        mode='max',
        patience=10,
        min_delta=0.001
    )
    callback_list.append(early_stop)
    
    checkpoint = plc.ModelCheckpoint(
        monitor='metric',
        dirpath=config.ckpt_dir,
        filename='{epoch:02d}-{metric:.3f}',
        save_top_k=-1,
        mode='max',
        save_last=True,
        every_n_epochs=1
    )
    callback_list.append(checkpoint)
    
    if config.lr_scheduler:
        lr_monitor = plc.LearningRateMonitor(logging_interval='step')
        callback_list.append(lr_monitor)
        
    return callback_list

def execute_training(config):
    pl.seed_everything(config.seed)
    
    training_model = MInterface(**vars(config))
    if config.ckpt_path:
        checkpoint_data = torch.load(config.ckpt_path, map_location='cpu')
        training_model.load_state_dict(checkpoint_data['state_dict'], strict=False)
        print(f"Loaded checkpoints from {config.ckpt_path}")

    dataset = DInterface(llm_tokenizer=training_model.llama_tokenizer, **vars(config))
    
    config.max_steps = len(dataset.trainset) * config.max_epochs // (config.accumulate_grad_batches * config.batch_size)
    
    tb_logger = TensorBoardLogger(save_dir='./log/', name=config.log_dir)
    config.callbacks = setup_callbacks(config)
    config.logger = tb_logger
    
    os.makedirs(config.ckpt_dir, exist_ok=True)
    
    training_system = Trainer.from_argparse_args(config)

    if config.auto_lr_find:
        learning_rate_finder = training_system.tuner.lr_find(
            model=training_model, 
            datamodule=dataset, 
            min_lr=1e-10, 
            max_lr=1e-3, 
            num_training=100
        )
        plot = learning_rate_finder.plot(suggest=True)
        plot.savefig("lr_finder.png")
        print("Saving to lr_finder.png")
        training_model.hparams.lr = learning_rate_finder.suggestion()

    if config.mode == 'train':
        training_system.fit(model=training_model, datamodule=dataset)
    else:
        training_system.test(model=training_model, datamodule=dataset)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    cmd_parser = ArgumentParser()

    cmd_parser.add_argument('--accelerator', default='gpu', type=str)
    cmd_parser.add_argument('--devices', default=-1, type=int)
    cmd_parser.add_argument('--precision', default='bf16', type=str)
    cmd_parser.add_argument('--amp_backend', default="native", type=str)
    cmd_parser.add_argument('--batch_size', default=8, type=int)
    cmd_parser.add_argument('--num_workers', default=8, type=int)
    cmd_parser.add_argument('--seed', default=1234, type=int)
    cmd_parser.add_argument('--lr', default=1e-3, type=float)
    cmd_parser.add_argument('--accumulate_grad_batches', default=8, type=int)
    cmd_parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    cmd_parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine'], type=str)
    cmd_parser.add_argument('--lr_decay_min_lr', default=1e-9, type=float)
    cmd_parser.add_argument('--lr_warmup_start_lr', default=1e-7, type=float)
    cmd_parser.add_argument('--load_best', action='store_true')
    cmd_parser.add_argument('--load_dir', default=None, type=str)
    cmd_parser.add_argument('--load_ver', default=None, type=str)
    cmd_parser.add_argument('--load_v_num', default=None, type=int)
    cmd_parser.add_argument('--dataset', default='movielens_data', type=str)
    cmd_parser.add_argument('--data_dir', default='data/ref/movielens1m', type=str)
    cmd_parser.add_argument('--model_name', default='mlp_projector', type=str)
    cmd_parser.add_argument('--loss', default='lm', type=str)
    cmd_parser.add_argument('--weight_decay', default=1e-5, type=float)
    cmd_parser.add_argument('--no_augment', action='store_true')
    cmd_parser.add_argument('--ckpt_dir', default='./checkpoints/', type=str)
    cmd_parser.add_argument('--log_dir', default='movielens_logs', type=str)
    cmd_parser.add_argument('--rec_size', default=64, type=int)
    cmd_parser.add_argument('--padding_item_id', default=1682, type=int)
    cmd_parser.add_argument('--llm_path', type=str)
    cmd_parser.add_argument('--rec_model_path', default='./rec_model/SASRec_ml1m.pt', type=str)
    cmd_parser.add_argument('--prompt_path', default='./prompt/movie/', type=str)
    cmd_parser.add_argument('--output_dir', default='./output/', type=str)
    cmd_parser.add_argument('--ckpt_path', type=str)
    cmd_parser.add_argument('--rec_embed', default="SASRec", choices=['SASRec', 'Caser','GRU'], type=str)
    cmd_parser.add_argument('--aug_prob', default=0.5, type=float)
    cmd_parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    cmd_parser.add_argument('--auto_lr_find', default=False, action='store_true')
    cmd_parser.add_argument('--metric', default='hr', choices=['hr'], type=str)
    cmd_parser.add_argument('--max_epochs', default=10, type=int)
    cmd_parser.add_argument('--save', default='part', choices=['part', 'all'], type=str)
    cmd_parser.add_argument('--cans_num', default=10, type=int)
    cmd_parser.add_argument('--llm_tuning', default='lora', choices=['lora', 'freeze','freeze_lora'], type=str)
    cmd_parser.add_argument('--peft_dir', default=None, type=str)
    cmd_parser.add_argument('--peft_config', default=None, type=str)
    cmd_parser.add_argument('--lora_r', default=8, type=float)
    cmd_parser.add_argument('--lora_alpha', default=32, type=float)
    cmd_parser.add_argument('--lora_dropout', default=0.1, type=float)

    args = cmd_parser.parse_args()
    
    dataset_padding = {
        'movielens': 1682,
        'steam': 3581,
        'lastfm': 4606
    }
    
    for dataset_name, padding_id in dataset_padding.items():
        if dataset_name in args.data_dir:
            args.padding_item_id = padding_id
            break

    execute_training(args)