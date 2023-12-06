import torch
from transformers import InstructBlipProcessor, AutoTokenizer

class BERTInstructBlipProcessor(InstructBlipProcessor):
    # TODO
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, qformer_tokenizer):
        super().__init__(image_processor, tokenizer, qformer_tokenizer)

    
    def to_bert(self, bert_name):
        # BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name) # TODO

class Processor():
    def __init__(self, processor):
        self.vision_processor = processor.image_processor
        self.tokenizer = processor.tokenizer
        self.qformer_tokenizer = processor.qformer_tokenizer
            
    def preprocess(self, sample):
        bbox = str([round(val, 3) for val in sample['bbox']])
        instruction = sample['instruction']

        input_image = self.vision_processor(sample['image'])
        pixel_values = input_image.pixel_values[0]

        input_text = self.tokenizer(instruction, padding='max_length', max_length=105)
        input_ids = input_text.input_ids 
        attention_mask = input_text.attention_mask

        output_text = self.qformer_tokenizer(instruction, padding='max_length', max_length=105)
        qformer_input_ids = output_text.input_ids
        qformer_attention_mask = output_text.attention_mask

        # gt
        decoder_text = self.tokenizer(bbox, padding='max_length', max_length=20)
        decoder_input_ids = decoder_text.input_ids
        decoder_attention_mask = decoder_text.attention_mask

        data = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'qformer_input_ids': qformer_input_ids,
            'qformer_attention_mask': qformer_attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'label_ids': decoder_input_ids,
        }

        return data
    
class QueryProcessor():
    def __init__(self, processor):
        self.vision_processor = processor.image_processor
        self.tokenizer = processor.tokenizer
        self.qformer_tokenizer = processor.qformer_tokenizer
            
    def preprocess(self, sample):
        instruction = sample['instruction']

        input_image = self.vision_processor(sample['image'])
        pixel_values = input_image.pixel_values[0]

        input_text = self.tokenizer(instruction, padding='max_length', max_length=105)
        input_ids = input_text.input_ids 
        attention_mask = input_text.attention_mask

        output_text = self.qformer_tokenizer(instruction, padding='max_length', max_length=105)
        qformer_input_ids = output_text.input_ids
        qformer_attention_mask = output_text.attention_mask

        labels = torch.tensor([round(val, 3) for val in sample['bbox']])

        data = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'qformer_input_ids': qformer_input_ids,
            'qformer_attention_mask': qformer_attention_mask,
            'labels': labels,
        }

        return data