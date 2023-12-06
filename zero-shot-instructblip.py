from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
from utils.eval import draw_bbox, save_bbox
from datasets import load_from_disk
import numpy as np
import random

# Seed for reproducibility
random.seed(34)

test_dataset = load_from_disk("/mnt/NAS/Andrew/InstructBlip/data/val")
# sample 10
# Randomly sample 20 numbers from 0 to 8548
sampled_numbers = random.sample(range(8549), 10)


# Path to your saved model checkpoint
# model_path = "Salesforce/instructblip-vicuna-7b" 
model_path = "Salesforce/instructblip-vicuna-13b" 
# model_path = "Salesforce/instructblip-flan-t5-xl"

# Load the model from your checkpoint
model = InstructBlipForConditionalGeneration.from_pretrained(model_path)
processor = InstructBlipProcessor.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for idx in sampled_numbers:
    sample = test_dataset[idx]
    image = sample['image']
    bbox = sample['bbox']
    width = sample['width']
    height = sample['height']
    category = sample['category_name']

    # image_file = 'tools/image.jpg'
    # image = Image.open("tools/image.jpg").convert("RGB")
    # category = 'person'
    # prompt = f"Where is {category} in the image? Return its bounding box coordinate in [center_x, center_y, width. height] format."
    prompt = f"Where is {category} in the image? Return its bounding box coordinate in [center_x, center_y, width. height] format."
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=512,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(result)
    # try:
    #     save_bbox(image, result, f"{image_file.split('.')[0]}-{category.replace(' ', '-')}-vicuna7b.png")
    # except:
    #     print('Not valid output')