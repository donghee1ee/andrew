import numpy as np

from datasets import load_dataset, Dataset
from tqdm import tqdm

def generate_instruction(category_name):
    instruction = f"Where is {category_name} in the image? Return its bounding box in [center_x, center_y, width. height] format.\n (center_x, center_y) is the center of the bounding box, and (width, height) is the length of the bounding box.\n All values are normalized from 0 to 1 relative to width and height of the image. The top-left corner of the image is the reference point (0, 0)."

    return instruction

def get_category_name(category):
    coco = {
        0: '__background__',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        12: 'stop sign',
        13: 'parking meter',
        14: 'bench',
        15: 'bird',
        16: 'cat',
        17: 'dog',
        18: 'horse',
        19: 'sheep',
        20: 'cow',
        21: 'elephant',
        22: 'bear',
        23: 'zebra',
        24: 'giraffe',
        25: 'backpack',
        26: 'umbrella',
        27: 'handbag',
        28: 'tie',
        29: 'suitcase',
        30: 'frisbee',
        31: 'skis',
        32: 'snowboard',
        33: 'sports ball',
        34: 'kite',
        35: 'baseball bat',
        36: 'baseball glove',
        37: 'skateboard',
        38: 'surfboard',
        39: 'tennis racket',
        40: 'bottle',
        41: 'wine glass',
        42: 'cup',
        43: 'fork',
        44: 'knife',
        45: 'spoon',
        46: 'bowl',
        47: 'banana',
        48: 'apple',
        49: 'sandwich',
        50: 'orange',
        51: 'broccoli',
        52: 'carrot',
        53: 'hot dog',
        54: 'pizza',
        55: 'donut',
        56: 'cake',
        57: 'chair',
        58: 'couch',
        59: 'potted plant',
        60: 'bed',
        61: 'dining table',
        62: 'toilet',
        63: 'tv',
        64: 'laptop',
        65: 'mouse',
        66: 'remote',
        67: 'keyboard',
        68: 'cell phone',
        69: 'microwave',
        70: 'oven',
        71: 'toaster',
        72: 'sink',
        73: 'refrigerator',
        74: 'book',
        75: 'clock',
        76: 'vase',
        77: 'scissors',
        78: 'teddy bear',
        79: 'hair drier',
        80: 'toothbrush'
            }
    
    return coco[category+1]

def convert_coco_to_yolo(coco_bbox, img_width, img_height):
    x, y, bbox_width, bbox_height = coco_bbox
    x = float(x)
    y = float(y)
    bbox_width = float(bbox_width)
    bbox_height = float(bbox_height)
    img_width = float(img_width)
    img_height = float(img_height)

    # Normalize the COCO coordinates
    x /= img_width
    y /= img_height
    bbox_width /= img_width
    bbox_height /= img_height

    # Calculate the center coordinates for YOLO
    center_x = x + (bbox_width / 2)
    center_y = y + (bbox_height / 2)

    return [center_x, center_y, bbox_width, bbox_height]

def convert_coco_to_yolo_2(coco_bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = coco_bbox
    x_min = float(x_min)
    y_min = float(y_min)
    x_max = float(x_max)
    y_max = float(y_max)
    img_width = float(img_width)
    img_height = float(img_height)

    x_min /= img_width
    y_min /= img_height
    x_max /= img_width
    y_max /= img_height

    # Calculate width and height
    width = x_max - x_min
    height = y_max - y_min

    # Calculate the center coordinates for YOLO
    center_x = x_min + (width / 2)
    center_y = y_min + (height / 2)

    return [center_x, center_y, width, height]

def generate_data(orig):
    temp_dataset = []
    image_id = []
    image = []
    width = []
    height = []
    category_num = []
    category_name = []
    instruction = []
    bbox= []

    for sample in tqdm(orig):
        objects = sample['objects']
        categories = objects['category']
        unique_categories, counts = np.unique(categories, return_counts=True)
        unique_category = unique_categories[counts == 1]
        temp_image_id = sample['image_id']
        temp_image = sample['image']
        temp_width = sample['width']
        temp_height = sample['height']

        for category in unique_category:
            idx = categories.index(category)
            
            image_id.append(temp_image_id)
            image.append(temp_image)
            width.append(temp_width)
            height.append(temp_height)
            category_num.append(category)
            category_name.append(get_category_name(category))
            instruction.append(generate_instruction(get_category_name(category)))
            bbox.append(convert_coco_to_yolo_2(objects['bbox'][idx], temp_width, temp_height))

            tqdm.write(f"count: {len(instruction)}, image_id {temp_image_id}, category: {get_category_name(category)}")
    
    dataset_data = Dataset.from_dict({
        "image_id" : image_id,
        "image" : image,
        "width" : width,
        "height" : height,
        "category_num" : category_num,
        "category_name" : category_name,
        "instruction" : instruction,
        "bbox" : bbox
    })

    # temp_dataset.append({
    #     "image_id" : image_id,
    #     "image" : image,
    #     "width" : width,
    #     "height" : height,
    #     "category_num" : category_num,
    #     "category_name" : category_name,
    #     "instruction" : instruction,
    #     "bbox" : bbox
    # })

    return dataset_data


orig_dataset = load_dataset("/mnt/NAS/Andrew/dataset/huggingface/detection-datasets___coco/")

orig_train = orig_dataset['train']
orig_val = orig_dataset['validation']

train = generate_data(orig_train)
val = generate_data(orig_val)

# train.save_to_disk("../data/train")
# val.save_to_disk("../data/val")