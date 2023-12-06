import logging
import os
import argparse
from typing import Dict
import random

import torch
# from torch.utils.data import DataLoader
from transformers import InstructBlipProcessor, TrainingArguments, Trainer

from datasets import load_from_disk
from model.modeling_instructblip import QueryT5InstructBlipForConditionalGeneration
from common.logger import setup_logger
from common.common import pretty_print
from common.compute_metrics import regression_compute_metrics
from model.processing_instructblip import QueryProcessor

# TODO
# 5. log only main process

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='t5_learnable')
    # /path/to/dataset
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/code/donghee/llava/data', help='path containing dataset')

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=1000, help='number of eval/test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--eval_steps', type=int, default=500, help='number of update steps between two evaluations')
    parser.add_argument('--logging_steps', type=int, default=100, help='number of steps between two logs')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')
    parser.add_argument('--save_all', type=bool, default=False, help='save all parameters including backbones')

    parser.add_argument(
        '--model_name', 
        type=str, 
        default='Salesforce/instructblip-flan-t5-xl',
        choices=['Salesforce/instructblip-flan-t5-xl', 'Salesforce/instructblip-flan-t5-xxl', 'Salesforce/instructblip-vicuna-7b'],
        help="Specifies the model to use. Choose from 'Salesforce/instructblip-flan-t5-xl' (default), "
            "'Salesforce/instructblip-flan-t5-xxl', or 'Salesforce/instructblip-vicuna-7b'."
    )

    parser.add_argument('--num_query', type=int, default=8, help='number of learnable query in decoder')

    args = parser.parse_args()
    
    args.output_dir= os.path.join("./outputs", args.project_name)
    args.logging_dir = os.path.join('./logs', args.project_name)
    if 't5' in args.model_name:
        args.decoder_only = False
    else:
        args.decoder_only = True

    return args

def train(args):
    model = QueryT5InstructBlipForConditionalGeneration.from_pretrained(args.model_name)
    model.reinit(args.save_all)

    train_dataset = load_from_disk(os.path.join(args.dataset_path, 'train')).select(range(1000))
    val_dataset = load_from_disk(os.path.join(args.dataset_path, 'val')).select(range(200))

    processor = InstructBlipProcessor.from_pretrained(args.model_name)
    processor_class = QueryProcessor(processor)

    processed_train_dataset = train_dataset.map(processor_class.preprocess, batched = False)
    processed_val_dataset = val_dataset.map(processor_class.preprocess, batched = False)

    processor.save_pretrained(os.path.join(args.output_dir, 'best'))
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps, # 500
        logging_dir=args.logging_dir,
        logging_strategy = 'steps',
        logging_steps=args.logging_steps,
        do_train=True,
        do_eval=True,
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        # include_inputs_for_metrics=True,
        # remove_unused_columns= False ## TODO
    )


    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_val_dataset,
        # data_collator = CustomDataCollator(tokenizer=processor.tokenizer, model=model)
        compute_metrics=regression_compute_metrics,
    )

    # Train the model
    trainer.train()

    # eval_result = trainer.evaluate(datasets['val'])
    # print("EVAL")
    # print(eval_result)

    # Save
    # processor.save_pretrained(os.path.join(args.output_dir, 'best'))
    model.save_pretrained_final(os.path.join(args.output_dir, 'best'))

    print("* Test start *")
    # test_results = trainer.evaluate(datasets['test'])
    # print(test_results)


if __name__ == '__main__':
    args = parse_args()
    setup_logger(args)

    ###
    args.batch_size = 16
    args.training_samples = 1000
    args.eval_samples = 100
    args.eval_steps = 20
    args.logging_steps = 5
    ###

    pretty_print(args)

    train(args)
