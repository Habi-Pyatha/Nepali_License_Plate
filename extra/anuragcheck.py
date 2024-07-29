# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:21:03 2023

@author: Dell
"""

from ultralytics import YOLO
import cv2
import numpy as np
from tkinter import filedialog
from classes import *

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_nepali_best.pt')
license_plate_number = YOLO('./models/license_plate_number_nepali_best.pt')

def open_image():
    file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

    if file_path:
        return file_path
    else:
        print("No file selected.")
        return None

# Get the image file path from the user
image_path = open_image()

# Check if a valid path is provided
if image_path:
    # Load the image
    frame = cv2.imread(image_path)

    # Resize the image for display
    img = cv2.resize(frame, (500, 500))
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)

    # Perform license plate detection
    license_plates = license_plate_detector(frame)[0]

    # Process the detected license plates
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

    license_plate_crop = frame[int(y1-10):int(y2), int(x1): int(x2), :]
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(license_plate_crop, (500, 500))
    cv2.imshow('Cropped License Plate', img)

    cv2.imshow('Grayscale', license_plate_crop_gray)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # Apply adaptive equalization
    image_adapteq = clahe.apply(license_plate_crop_gray)
    cv2.imshow('Equalized', image_adapteq)
    cv2.waitKey(0)

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

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert the list of detections to a numpy array for easier manipulation
    detections_array = np.array(detections)

    # Sort the detections by y-coordinate (column index 1) and then x-coordinate (column index 0)
    sorted_detections_array = detections_array[np.lexsort((detections_array[:, 0], detections_array[:, 1]))]

    # Convert the sorted numpy array back to a list of detections
    sorted_detections = sorted_detections_array.tolist()

    # Separate the list into two halves
    list1 = sorted_detections[:len(sorted_detections)//2]
    list2 = sorted_detections[len(sorted_detections)//2:]

    # Process each half separately
    corrected_numbers = []

    for lst in [list1, list2]:
        # Convert the list of detections to a numpy array for easier manipulation
        detections_array = np.array(lst)

        # Sort the detections by x-coordinate (column index 0) only
        sorted_detections_array = detections_array[np.argsort(detections_array[:, 0])]

        # Convert the sorted numpy array back to a list of detections
        sorted_detections = sorted_detections_array.tolist()

        # Concatenate the corrected numbers from both halves
        corrected_numbers.extend([get_object_name(detection[4]) for detection in sorted_detections])

    # Create a new window and display the final result string
    result_string = ''.join(corrected_numbers)

    # Format the result string according to the specified pattern
    province = result_string[:2]
    lot_number = result_string[2:4]
    vehicle_type = result_string[4:6]
    plate_number = result_string[6:]

    # Create an image to display both the result string and the pattern
    pattern_image = np.ones((300, 600, 3), np.uint8) * 255  # White background
    cv2.putText(pattern_image, f"Pattern:", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(pattern_image, f"Province: {province}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(pattern_image, f"Lot No.: {lot_number}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(pattern_image, f"Vehicle Type: {vehicle_type}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(pattern_image, f"Plate Number: {plate_number}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(pattern_image, f"Result String: {result_string}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.namedWindow('Final Result', cv2.WINDOW_NORMAL)
    cv2.imshow('Final Result', pattern_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Stored in Variable =", result_string)
