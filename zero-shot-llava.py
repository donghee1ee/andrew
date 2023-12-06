from utils.eval import calculate_iou, draw_bbox, save_bbox
from model.LLaVA.llava.model.builder import load_pretrained_model
from model.LLaVA.llava.mm_utils import get_model_name_from_path
from model.LLaVA.llava.eval.run_llava import eval_model
from PIL import Image, ImageDraw

model_path = "liuhaotian/llava-v1.5-13b"

category = 'person'
prompt = f"Where is {category} in the image? Return its bounding box in [center_x, center_y, width. height] format."
image_file = "tools/image.jpg"
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature" : 0.2,
    "top_p" : 0.7,
    "num_beams" : 1,
    "max_new_tokens": 1024
})()

result = eval_model(args)

print(result)
image = Image.open(image_file).convert('RGB')
draw_bbox(image, result)
save_bbox(image, result, f"{image_file.split('.')[0]}-{category.replace(' ', '-')}-llava.png")