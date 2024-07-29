import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from classid import *

# Load models
license_plate_detector = YOLO('./models/license_plate_nepali_best.pt')


def process_image(frame):
    
    license_plates = license_plate_detector(frame)[0]
    license_plate_number = YOLO('./models/license_plate_number_nepali_best.pt')
    # Process the detected license plates
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
    
    license_plate_crop = frame[int(y1-10):int(y2), int(x1): int(x2), :]
    
    img = cv2.resize(license_plate_crop, (500, 500))
    
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

    # Format province name
    if province == 'Ba':
        province = "Bagmati"

    # Format vehicle type
    vehicle_type_name = "Two Wheeler" if vehicle_type == 'Pa' else "Four Wheeler"

    # Create a white background image
    pattern_image = np.ones((800, 800, 3), np.uint8) * 255  # White background

    # Define text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    font_color = (0, 0, 0)  # Black color for text

    # Add text to the image
    cv2.putText(pattern_image, "Pattern:", (50, 50), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(pattern_image, f"Province: {province}", (50, 100), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(pattern_image, f"Lot No.: {lot_number}", (50, 150), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(pattern_image, f"Vehicle Type: {vehicle_type_name}", (50, 200), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(pattern_image, f"Plate Number: {plate_number}", (50, 250), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(pattern_image, f"Result String: {result_string}", (50, 300), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return image, result_string, pattern_image

def get_object_name(class_id):
    object_names = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'Ba', 11: 'Cha', 12: 'Ga', 13: 'Ja', 14: 'Jha', 15: 'Ka', 16: 'Kha', 17: 'Lu', 18: 'Pa'
    }
    return object_names.get(class_id, 'Unknown')

# Streamlit app
st.title("License Plate Recognition")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    #image1 = uploaded_file
    image, result_string, pattern_image = process_image(frame)

    # Resize the images to have the same height (assuming images have the same aspect ratio)
    max_height = max(image.shape[0], pattern_image.shape[0])
    image = cv2.resize(image, (int(image.shape[1] * max_height / image.shape[0]), max_height))
    pattern_image = cv2.resize(pattern_image, (int(pattern_image.shape[1] * max_height / pattern_image.shape[0]), max_height))

    # Concatenate the images horizontally
    images_combined = np.hstack([ image, pattern_image])
    st.image(uploaded_file, caption='Input Image', use_column_width=True)
    st.image(images_combined, caption='Plate Details', use_column_width=True)
    
    st.write("Result String:", result_string)


st.write("Developed By: Anurag Poudel, Avishek Hada, Habi Pyatha ,Sujan Dhoj Karki")