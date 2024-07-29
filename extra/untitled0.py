# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:56:10 2024

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
from tkinter import filedialog, Tk, Label, Button
from PIL import Image, ImageTk

# Load models
# coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_nepali_best.pt')
license_plate_number = YOLO('./models/license_plate_number_nepali_best.pt')

# Initialize Tkinter
root = Tk()
root.title("License Plate Recognition")
root.geometry("800x800")


def get_object_name(class_id):
    # Your implementation remains the same
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
    # Your implementation remains the same
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
def process_image(image_path):
    frame = cv2.imread(image_path)
    license_plate_number = YOLO('./models/license_plate_number_nepali_best.pt')#load model
    # Perform license plate detection
    license_plates = license_plate_detector(frame)[0]

    # Process the detected license plates
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

    license_plate_crop = frame[int(y1-10):int(y2), int(x1): int(x2), :]
    license_plates_numbers = license_plate_number(license_plate_crop)[0]

    # Your processing and drawing code remains the same
    numbers=[]
    detections=[]
    image = license_plate_crop
    # print("hello=",license_plates_numbers)
    for license_plate_number in license_plates_numbers.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate_number
                # print(license_plate_number)
                # print("class id",class_id)
                # numbers.append([class_id])
                # print(score)
                # if score>50.0:
                
            
                detections.append([x1, y1, x2, y2, class_id])
                
                top_left = (x1, y1)
                bottom_right = (x2, y2)
                
                object_name = get_object_name(class_id)        

                # Call the function to add class ID to the object's name
                add_class_id(image, class_id, object_name, top_left=top_left, bottom_right=bottom_right)
    detections_array = np.array(detections)


    # Sort the detections by y-coordinate (column index 1) and then x-coordinate (column index 0)
    sorted_detections_array = detections_array[np.lexsort((detections_array[:, 0], detections_array[:, 1]))]

    # Convert the sorted numpy array back to a list of detections
    sorted_detections = sorted_detections_array.tolist()
    # print("diff")
    # Print the sorted detections
    for detection in sorted_detections:
        x1, y1, x2, y2, class_id = detection
        
    list1 = sorted_detections[:len(sorted_detections)//2]
    list2 = sorted_detections[len(sorted_detections)//2:]



    # Convert the list of detections to a numpy array for easier manipulation
    detections_array = np.array(list1)

    # Sort the detections by x-coordinate (column index 0) only
    sorted_detections_array = detections_array[np.argsort(detections_array[:, 0])]

    # Convert the sorted numpy array back to a list of detections
    sorted_detections = sorted_detections_array.tolist()
    corrected_numbers=[]
    # Print the sorted detections
    for detection in sorted_detections:
        x1, y1, x2, y2, class_id = detection
        
        object_name = get_object_name(class_id)
        corrected_numbers.append([object_name])
        

    result_string= ''.join(''.join(row) for row in corrected_numbers)
    # print(result_string)
        
    detections_array = np.array(list2)

    # Sort the detections by x-coordinate (column index 0) only
    sorted_detections_array = detections_array[np.argsort(detections_array[:, 0])]

    # Convert the sorted numpy array back to a list of detections
    sorted_detections = sorted_detections_array.tolist()

    # Print the sorted detections
    for detection in sorted_detections:
        x1, y1, x2, y2, class_id = detection
        
        object_name = get_object_name(class_id)
        corrected_numbers.append([object_name])
        

    result_string= ''.join(''.join(row) for row in corrected_numbers)
    print("Stored in Variable=",result_string)
    # Convert the resulting image to RGB format for displaying in Tkinter
    image_rgb = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    image_tk = ImageTk.PhotoImage(image_pil)

    # Create a label to display the image
    img_label = Label(root, image=image_tk)
    img_label.image = image_tk
    img_label.pack()

    # Display the final result string
    result_label = Label(root, text=f"Result String: {result_string}")
    result_label.pack()

def select_image():
    # Open a file dialog to select an image
    image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    # Check if the user selected a file
    if not image_path:
        print("No file selected. Exiting.")
        return

    # Process the selected image
    process_image(image_path)

# Create a button to select an image
select_button = Button(root, text="Select Image", command=select_image)
select_button.pack()

# Start the Tkinter event loop
root.mainloop()
