import argparse
import os

from transformers import InstructBlipProcessor, TrainingArguments, Trainer
from datasets import load_dataset

from model.modeling_instructblip import FreezeInstructBlipForConditionalGeneration

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='andrew')
    # /path/to/Recipe1M/dataset
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/code/donghee/inversecooking/data', help='path containing Recipe1M dataset')

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=-1, help='number of eval/test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--eval_steps', type=int, default=500, help='number of update steps between two evaluations')
    parser.add_argument('--logging_steps', type=int, default=100, help='number of steps between two logs')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')

    parser.add_argument(
        '--model_name', 
        type=str, 
        default='Salesforce/instructblip-flan-t5-xl',
        choices=['Salesforce/instructblip-flan-t5-xl', 'Salesforce/instructblip-flan-t5-xxl', 'Salesforce/instructblip-vicuna-7b'],
        help="Specifies the model to use. Choose from 'Salesforce/instructblip-flan-t5-xl' (default), "
            "'Salesforce/instructblip-flan-t5-xxl', or 'Salesforce/instructblip-vicuna-7b'."
    )

    args = parser.parse_args()
    
    args.output_dir= os.path.join("./outputs", args.project_name)
    args.logging_dir = os.path.join('./logs', args.project_name)
    if 't5' in args.model_name:
        args.decoder_only = False
    else:
        args.decoder_only = True

    return args

class Processor():
    def __init__(self, processor):
        self.vision_processor = processor.image_processor
        self.tokenizer = processor.tokenizer
        self.qformer_tokenizer = processor.qformer_tokenizer
            
    def preprocess(self, example):
        # TODO better optimization
        
        ### pre_preprocess
        pixel_values = self.vision_processor(example['image']).pixel_values
        bbox = example['objects']['bbox']
        starting = 'Bounding box is located at '
        temp = starting
        for i, box in enumerate(bbox):
            temp += (str(box))
            if i != len(bbox)-1:
                temp += ' and '
        bbox = temp
        prompt = 'Find any objects and return their bounding boxes in[x_min, y_min, x_max, y_max] format'
        
        input_text = self.tokenizer(prompt, padding='max_length', max_length=128)
        input_ids = input_text.input_ids 
        attention_mask = input_text.attention_mask

        output_text = self.qformer_tokenizer(prompt, padding='max_length', max_length=128)
        qformer_input_ids = output_text.input_ids
        qformer_attention_mask = output_text.attention_mask

        decoder_text = self.tokenizer(bbox, padding='max_length', max_length=128)
        decoder_input_ids = decoder_text.input_ids
        decoder_attention_mask = decoder_text.attention_mask

        sample = {
            'pixel_value': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'qformer_input_ids': qformer_input_ids,
            'qformer_attention_mask': qformer_attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'label_ids': decoder_input_ids,
        }

        return sample

def andrew():
    dataset = load_dataset("detection-datasets/coco")
    processor = InstructBlipProcessor.from_pretrained(args.model_name)
    processor_class = Processor(processor)
    processed_dataset = dataset.map(processor_class.preprocess, batched = False)
    processor.save_pretrained(os.path.join(args.output_dir, 'best'))

    model = FreezeInstructBlipForConditionalGeneration.from_pretrained(args.model_name)

    # training arg
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
        metric_for_best_model='loss', ## TODO
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        # include_inputs_for_metrics=True,
        # remove_unused_columns= False 
    )

    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['val'],
        # data_collator = CustomDataCollator(tokenizer=processor.tokenizer, model=model)
        # compute_metrics=eval_metrics.compute_metrics, ## TODO - regression / generation 
        # data_collator=data_collator
        # metric_class = eval_metrics,
    )

    # Train the model
    trainer.train()

    # Save
    model.save_pretrained_final(os.path.join(args.output_dir, 'best'))

    print("* Test start *")
    test_results = trainer.evaluate(processed_dataset['val'])
    print(test_results)


if __name__ == '__main__':
    args = argparse()
    ##
    args.epoch = 10
    args.batch_size = 16
    ##
    andrew(args)