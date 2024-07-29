# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:48:18 2023

@author: Dell
"""

import torch
from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm

# Set the path to your image
image_path = "./eight.jpg"

# Set the path to the YOLOv8 weights file
weights_path = "./license_plate_number_nepali_best.pt"

# Load the YOLOv8 model
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)

# Load the image
img = Image.open(image_path)

# Run the YOLOv8 model on the image
results = model(img)

# Extract bounding box coordinates and class IDs
boxes = results.xyxy[0][:, :4].cpu().numpy()
class_ids = results.xyxy[0][:, 5].cpu().numpy()

# Sort detections by y-coordinate and then x-coordinate
sorted_indices = boxes[:, 1].argsort()

# Create an output image with class IDs appended from top left to bottom right
draw = ImageDraw.Draw(img)
for i in tqdm(sorted_indices, desc="Drawing bounding boxes"):
    box = boxes[i].astype(int)
    class_id = int(class_ids[i])
    class_name = f"Class {class_id}"

    # Draw the bounding box
    draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=2)

    # Draw the class ID
    draw.text((box[0], box[1]), class_name, fill="red")

# Save or display the output image
output_path = "path/to/your/output/image_with_class_ids.jpg"
img.save(output_path)
img.show()