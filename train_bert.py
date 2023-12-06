import logging
import os
import argparse

import torch
from transformers import InstructBlipProcessor, TrainingArguments, Trainer, InstructBlipConfig

from data.dataset import load_datasets
from data.utils import Vocabulary
from model.modeling_instructblip import FreezeInstructBlipForConditionalGeneration, BERTInstructBlipForConditionalGeneration
from model.processing_instructblip import BERTInstructBlipProcessor
from common.logger import setup_logger
from common.common import pretty_print
from common.compute_metrics import compute_metrics

# TODO
# 5. log only main process /

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='BERT_vit_train_pretrained_qformer')
    # /path/to/dataset
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/code/donghee/inversecooking/data', help='path containing Recipe1M dataset')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--eval_steps', type=int, default=500) 
    parser.add_argument('--logging_steps', type=int, default=50) 
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=1500, help='number of eval/test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')
    parser.add_argument('--load_from_cache_file', type=bool, default=True, help='load dataset from huggingface cache')
    parser.add_argument('--train_llm', type=bool, default=False, help='train llm backbone')
    parser.add_argument('--train_vit', type=bool, default=False, help='train ViT')

    parser.add_argument(
        '--model_name', 
        type=str, 
        default='Salesforce/instructblip-flan-t5-xl',
        choices=['Salesforce/instructblip-flan-t5-xl', 'Salesforce/instructblip-flan-t5-xxl', 'Salesforce/instructblip-vicuna-7b'],
        help="Specifies the model to use. Choose from 'Salesforce/instructblip-flan-t5-xl' (default), "
            "'Salesforce/instructblip-flan-t5-xxl', or 'Salesforce/instructblip-vicuna-7b'."
    )
    parser.add_argument(
        '--bert_name', 
        type=str, 
        default='bert-large-uncased',
        choices=['bert-large-uncased', 'bert-base-uncased'],
        help="Specifies the BERT model to use. Choose from 'bert-large-uncased' (default), "
            "or 'bert-base-uncased'."
    )
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    args = parser.parse_args()

    if args.bert_name is not None:
        args.encoder_only = True
    
    args.output_dir= os.path.join("./outputs", args.project_name)
    args.logging_dir = os.path.join('./logs', args.project_name)
    if 't5' in args.model_name:
        args.decoder_only = False
    else:
        args.decoder_only = True

    return args

def train(args):

    model = BERTInstructBlipForConditionalGeneration(args.bert_name, args.train_llm, args.train_vit) # TODO better way
    processor = BERTInstructBlipProcessor.from_pretrained(args.model_name) # TODO - better way
    processor.to_bert(args.bert_name) # TODO

    datasets = load_datasets( 
        processor=processor, 
        data_dir=args.dataset_path, 
        training_samples=args.training_samples,
        eval_samples=args.eval_samples, 
        pre_map=args.pre_map,
        decoder_only=args.decoder_only,
        encoder_only = args.encoder_only,
        load_from_cache_file = args.load_from_cache_file
    )
    
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
        save_steps = args.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        compute_metrics=compute_metrics,
        # data_collator = CustomDataCollator(tokenizer=tokenizer, model=model)
        # callbacks=[PrinterCallback]
    )

    # Train the model
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # eval_result = trainer.evaluate()
    # print("EVAL")
    # print(eval_result)


    # Save
    processor.save_pretrained(os.path.join(args.output_dir, 'best')) # TODO not trained..why save..
    model.save_pretrained_final(os.path.join(args.output_dir, 'best'))

    print("* Test start *")
    test_results = trainer.evaluate(datasets['test'])
    print(test_results)


if __name__ == '__main__':
    args = parse_args()
    setup_logger(args)

    ####
    # args.training_samples = 150
    args.epochs = 100
    args.train_llm = False
    args.resume_from_checkpoint = '/nfs_share2/code/donghee/instructBlip/outputs/BERT/checkpoint-32240'
    args.batch_size = 16
    args.train_vit = True
    # args.eval_steps = 10
    ####

    pretty_print(args)

    train(args)
