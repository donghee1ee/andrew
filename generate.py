from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests

# Path to your saved model checkpoint
model_path = "Salesforce/instructblip-vicuna-7b" 
# model_path = "Salesforce/instructblip-vicuna-13b" 
# Salesforce/instructblip-flan-t5-xl

# Load the model from your checkpoint
model = InstructBlipForConditionalGeneration.from_pretrained(model_path)
processor = InstructBlipProcessor.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

url = "tools/image-3.jpg"
image = Image.open("tools/image.jpg").convert("RGB")
prompt = "Where is cat in the image? Return its bounding box values in [center_x, center_y, width, height] format. All values are normalized from 0 to 1 relative to width and height of the image. The top-left corner of the image is the reference point (0, 0)."
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
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)