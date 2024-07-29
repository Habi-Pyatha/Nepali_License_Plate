# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:26:40 2024

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:21:03 2023

@author: Dell
"""

from ultralytics import YOLO
import cv2
import numpy as np







# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_nepali_best.pt')

license_plate_number = YOLO('./models/license_plate_number_nepali_best.pt')

# license_plate_number = YOLO('./models/preprocessed_numbers.pt')
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
        return 'Cha'
    elif class_id == 12.0:
        return 'Ga'
    elif class_id == 13.0:
        return 'Ja'
    elif class_id == 14.0:
        return 'Jha'
    elif class_id == 15.0:
        return 'Ka'
    elif class_id == 16.0:
        return 'Kha'
    elif class_id == 17.0:
        return 'Lu'
    elif class_id == 18.0:
        return 'Pa'
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
# cap = cv2.VideoCapture('./test.mp4')
frame=cv2.imread("./scho.png")
img=cv2.resize(frame,(500,500))
cv2.imshow('img',img)
cv2.waitKey(0)
license_plates = license_plate_detector(frame)[0]
for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            # print(license_plate)
license_plate_crop = frame[int(y1-10):int(y2), int(x1): int(x2), :]
# license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
# license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_RGB2GRAY)

# print(license_plate_crop_gray.shape)

# license_plate_crop_gray = license_plate_crop_gray.flatten()

# _,license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
img=cv2.resize(license_plate_crop,(500,500))
cv2.imshow('crop',img)
# cv2.waitKey(0)
# cv2.imshow('grayscale',license_plate_crop_gray)
# # cv2.waitKey(0)
# # cv2.imshow('thresh',license_plate_crop_thresh)
# # cv2.waitKey(0)
# # Create a CLAHE object
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# # Apply adaptive equalization
# image_adapteq = clahe.apply(license_plate_crop_gray)
# cv2.imshow('equalized',image_adapteq)
# cv2.waitKey(0)
# print(image_adapteq.shape)
# Assuming 'image_adapteq' is your 4D array (shape: (1, 3, height, width))
# Select the first channel

# image_adapteq = image_adapteq.transpose((0, 3, 1, 2))
# image_adapteq = image_adapteq.transpose((0, 2, 3, 1))
# import numpy as np

# Assuming image_adapteq has shape (1, 3, 800, 384)
# image_adapteq = np.moveaxis(image_adapteq, [0, 1, 2, 3], [0, 2, 3, 1])


# Create a mask for regions where adaptive equalization will be applied
# blur_mask = cv2.GaussianBlur(license_plate_crop_gray, (15, 15), 0)
# cv2.imshow('blur_mask',blur_mask)

# # Combine the original image and the equalized image using the mask
# result = np.where(blur_mask > 200, image_adapteq, license_plate_crop_gray)
# cv2.imshow('result',result)

license_plates_numbers = license_plate_number(license_plate_crop)[0]
# license_plates_numbers = license_plate_number(image_adapteq)[0]

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


#start test
# Apply adaptive equalization
# print(detections)
#end test





# Display the image
# image=cv2.resize(image,(500,500))
cv2.imshow("Image with Class ID", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Example list of detections (x1, y1, x2, y2, class_id)
# detections = [(10, 50, 30, 70, 1), (20, 30, 40, 50, 2), (5, 60, 25, 80, 3)]

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

