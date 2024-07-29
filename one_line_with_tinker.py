# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:01:06 2024

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:21:03 2023

@author: Habi Pyatha
"""

from ultralytics import YOLO
import cv2
import numpy as np
from tkinter import filedialog
import tkinter as tk
# from PIL import Image, ImageTk  # Added import statements
# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_nepali_best.pt')

# license_plate_number = YOLO('./models/license_plate_number_nepali_best.pt')
license_plate_number = YOLO('./models/preprocessed_numbers.pt')
def get_object_name(class_id):
    if class_id == 0.0:
        return '0'
    elif class_id == 1.0:
        return '1'
    elif class_id == 2.0:
        return '2'
    elif class_id == 3.0:
        return '3'
    elif class_id == 4.0:
        return '4'
    elif class_id == 5.0:
        return '5'
    elif class_id == 6.0:
        return '6'
    elif class_id == 7.0:
        return '7'
    elif class_id == 8.0:
        return '8'
    elif class_id == 9.0:
        return '9'
    elif class_id == 10.0:
        return 'Ba'
    elif class_id == 11.0:
        return 'Cha'#four
    elif class_id == 12.0:
        return 'Ga'#four
    elif class_id == 13.0:
        return 'Ja'#four
    elif class_id == 14.0:
        return 'Jha'#four
    elif class_id == 15.0:
        return 'Ka'#four
    elif class_id == 16.0:
        return 'Kha'#four
    elif class_id == 17.0:
        return 'Lu'#four
    elif class_id == 18.0:
        return 'Pa'#two
    else:
        return 'Unknown'

def add_class_id(image, class_id, object_name, top_left, bottom_right):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 5
    font_color = (0, 255, 255)  # White color for text

    # Convert the coordinates to integers
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    # Draw the bounding box on the image
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Create the text to be displayed
    # text = f"{object_name} (Class {class_id})"
    text=f"{object_name}"

    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate the position for the text
    text_position = (top_left[0]+50, top_left[1] +20)

    # Draw the text on the image
    cv2.putText(image, text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

# load video
# Create a Tkinter root window (it will not be shown)
root = tk.Tk()
root.title("License Plate Recognition")
root.withdraw()


# Ask the user to choose an image file using a file dialog
image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

# Check if the user selected a file
if not image_path:
    print("No file selected. Exiting.")
    exit()
   
    
    

# Load the selected image
frame = cv2.imread(image_path)
# cap = cv2.VideoCapture('./test.mp4')
# frame=cv2.imread("./eight.jpg")
img=cv2.resize(frame,(500,500))
# cv2.imshow('img',img)
license_plate_number = YOLO('./models/license_plate_number_nepali_best.pt')
# cv2.waitKey(0)
image1=img
 # Perform license plate detection
license_plates = license_plate_detector(frame)[0]

# Process the detected license plates
for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate

license_plate_crop = frame[int(y1-10):int(y2), int(x1): int(x2), :]
# license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

img = cv2.resize(license_plate_crop, (500, 500))
# cv2.imshow('Cropped License Plate', img)

# cv2.imshow('Grayscale', license_plate_crop_gray)

# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# # Apply adaptive equalization
# image_adapteq = clahe.apply(license_plate_crop_gray)
# cv2.imshow('Equalized', image_adapteq)
# cv2.waitKey(0)

license_plates_numbers = license_plate_number(license_plate_crop)[0]

numbers = []
detections = []
image = license_plate_crop

for license_plate_number in license_plates_numbers.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate_number
    detections.append([x1, y1, x2, y2, class_id])
    top_left = (x1, y1)
    bottom_right = (x2, y2)
    object_name = get_object_name(class_id)
    add_class_id(image, class_id, object_name, top_left=top_left, bottom_right=bottom_right)


# cv2.imshow("Image with Class ID", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Convert the list of detections to a numpy array for easier manipulation
detections_array = np.array(detections)

# Sort the detections by y-coordinate (column index 1) and then x-coordinate (column index 0)



sorted_detections_array = detections_array[np.argsort(detections_array[:, 0])]

corrected_numbers=[]
# Convert the sorted numpy array back to a list of detections
sorted_detections = sorted_detections_array.tolist()

# Print the sorted detections
for detection in sorted_detections:
    x1, y1, x2, y2, class_id = detection
    
    object_name = get_object_name(class_id)
    corrected_numbers.append([object_name])
    
result_string= ''.join(''.join(row) for row in corrected_numbers)
# result_string=corrected_numbers
print("Stored in Variable=",result_string)

# Format the result string according to the specified pattern
province = result_string[:2]
lot_number = result_string[2:4]
vehicle_type = result_string[4:6]
plate_number = result_string[6:]
import re

def extract_numbers(s):
    numbers = re.findall(r'\d+', s)
    return ''.join(numbers)

# Example usage:
# s = "abc123def456ghi789"
# numbers = extract_numbers(s)
# print(numbers)
plate_number=extract_numbers(plate_number)
#for province
if province=='Ba':
    province="Bagmati" 
#for two or four wheeler
if vehicle_type=='Pa':
    vehicle_type_name="Two Wheeler" 
else:
    vehicle_type_name="Four Wheeler"

# Create an image to display both the result string and the pattern
# pattern_image = np.ones((600, 600, 3), np.uint8) * 255  # White background
# cv2.putText(pattern_image, f"Pattern:", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
# cv2.putText(pattern_image, f"Province: {province}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
# cv2.putText(pattern_image, f"Lot No.: {lot_number}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
# cv2.putText(pattern_image, f"Vehicle Type: {vehicle_type_name}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
# cv2.putText(pattern_image, f"Plate Number: {plate_number}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
# cv2.putText(pattern_image, f"Result String: {result_string}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

# cv2.namedWindow('Final Result', cv2.WINDOW_NORMAL)
# cv2.imshow('Final Result', pattern_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#

# Create a white background image
pattern_image = np.ones((800, 800, 3), np.uint8) * 255  # White background

# Define text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
font_thickness = 2
font_color = (0, 0, 0)  # Black color for text

# Define text content
# province = "Province Name"
# lot_number = "Lot Number"
# vehicle_type_name = "Vehicle Type"
# plate_number = "Plate Number"
# result_string = "Result String"

# Add text to the image
cv2.putText(pattern_image, f"Pattern:", (50, 50), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
cv2.putText(pattern_image, f"Province: {province}", (50, 100), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
cv2.putText(pattern_image, f"Lot No.: {lot_number}", (50, 150), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
cv2.putText(pattern_image, f"Vehicle Type: {vehicle_type_name}", (50, 200), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
cv2.putText(pattern_image, f"Plate Number: {plate_number}", (50, 250), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
cv2.putText(pattern_image, f"Result String: {result_string}", (50, 300), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

# Display the resulting image
# cv2.namedWindow('Final Result', cv2.WINDOW_NORMAL)
# cv2.imshow('Final Result', pattern_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
print("Stored in Variable =", result_string)
# three images side by side
# Load the images
# image1 = frame# mentioned above
image2 = image
image3 = pattern_image

# Resize the images to have the same height (assuming images have the same aspect ratio)
max_height = max(image1.shape[0], image2.shape[0], image3.shape[0])
image1 = cv2.resize(image1, (int(image1.shape[1] * max_height / image1.shape[0]), max_height))
image2 = cv2.resize(image2, (int(image2.shape[1] * max_height / image2.shape[0]), max_height))
image3 = cv2.resize(image3, (int(image3.shape[1] * max_height / image3.shape[0]), max_height))

# Concatenate the images horizontally
images_combined = cv2.hconcat([image1, image2, image3])

# Display the combined image
images_combined=cv2.resize(images_combined,(1500,800))
cv2.imshow("Combined Images", images_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

