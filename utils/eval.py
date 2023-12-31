import numpy as np
import cv2
import matplotlib.pyplot as plt
import ast
from PIL import Image

import numpy as np

def calculate_iou_batch(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) for batches of bounding boxes.
    The bounding boxes are expected in [center_x, center_y, width, height] format.
    bbox1 and bbox2 are arrays of bounding boxes, where each row is a bounding box.
    """
    # Convert from center format to corner format
    bbox1 = bbox_center_to_corners_batch(bbox1)
    bbox2 = bbox_center_to_corners_batch(bbox2)

    # Determine the coordinates of the intersection rectangles
    x_left = np.maximum(bbox1[:, 0:1], bbox2[:, 0:1])
    y_top = np.maximum(bbox1[:, 1:2], bbox2[:, 1:2])
    x_right = np.minimum(bbox1[:, 2:3], bbox2[:, 2:3])
    y_bottom = np.minimum(bbox1[:, 3:4], bbox2[:, 3:4])

    # Calculate the area of the intersection rectangles
    intersection_area = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)

    # Calculate the area of both bounding boxes
    bbox1_area = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    bbox2_area = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])

    # Calculate IoU
    iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    # Handle cases where there is no overlap
    no_overlap = np.logical_or(x_right < x_left, y_bottom < y_top)
    iou[no_overlap] = 0.0

    return iou

def bbox_center_to_corners_batch(bboxes):
    """
    Convert bounding boxes from [center_x, center_y, width, height] to 
    [x_min, y_min, x_max, y_max] format for a batch of bounding boxes.
    """
    x_min = bboxes[:, 0] - bboxes[:, 2] / 2
    y_min = bboxes[:, 1] - bboxes[:, 3] / 2
    x_max = bboxes[:, 0] + bboxes[:, 2] / 2
    y_max = bboxes[:, 1] + bboxes[:, 3] / 2
    return np.stack([x_min, y_min, x_max, y_max], axis=1)


def bbox_center_to_corners(bbox):
    """
    Convert a bounding box from [center_x, center_y, width, height] format 
    to [x_min, y_min, x_max, y_max] format.
    """
    center_x, center_y, width, height = bbox
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2
    return [x_min, y_min, x_max, y_max]

def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    The bounding boxes are expected in [center_x, center_y, width, height] format.
    """
    bbox1 = bbox_center_to_corners(bbox1)
    bbox2 = bbox_center_to_corners(bbox2)

    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # Check if there is no overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of the intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate the IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou


def draw_bbox(image, bbox):
    """
    Draw YOLO bounding box on the image.
    
    :param image: The image (as a numpy array) on which to draw.
    :param bbox: The bounding box in YOLO format [center_x, center_y, width, height] (normalized).
    """
    image = np.array(image)
    img_height, img_width = image.shape[:2]
    if type(bbox) != 'str':
        bbox = ast.literal_eval(bbox)

    # Denormalize the coordinates
    center_x, center_y, width, height = bbox
    center_x *= img_width
    center_y *= img_height
    width *= img_width
    height *= img_height

    # Convert to top-left corner format
    x_min = int(center_x - width / 2)
    y_min = int(center_y - height / 2)


    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_min + int(width), y_min + int(height)), (255, 0, 0), 2)

    # Display the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def save_bbox(image, bbox, save_image):    
    image = np.array(image)
    img_height, img_width = image.shape[:2]
    if type(bbox) == 'str':
        bbox = ast.literal_eval(bbox)

    # Denormalize the coordinates
    center_x, center_y, width, height = bbox
    center_x *= img_width
    center_y *= img_height
    width *= img_width
    height *= img_height

    # Convert to top-left corner format
    x_min = int(center_x - width / 2)
    y_min = int(center_y - height / 2)


    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_min + int(width), y_min + int(height)), (255, 0, 0), 2)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_image, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))