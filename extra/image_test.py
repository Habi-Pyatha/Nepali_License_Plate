# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:21:03 2023

@author: Dell
"""

from ultralytics import YOLO
import cv2
import numpy as np
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_nepali_best.pt')

license_plate_number = YOLO('./models/license_plate_number_nepali_best.pt')

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
frame=cv2.imread("./eight.jpg")
img=cv2.resize(frame,(500,500))
cv2.imshow('img',img)
cv2.waitKey(0)
license_plates = license_plate_detector(frame)[0]
for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            # print(license_plate)
license_plate_crop = frame[int(y1-10):int(y2), int(x1): int(x2), :]
# Resize the image to a compatible size
# resized_license_plate_crop = cv2.resize(license_plate_crop, (416, 416))

license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
# # Convert image to BGR if it's in RGB format
# license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

_,license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
img=cv2.resize(license_plate_crop,(500,500))
cv2.imshow('img',img)
cv2.waitKey(0)


img1=cv2.resize(license_plate_crop_gray,(500,500))
cv2.imshow('img',img1)
cv2.waitKey(0)
# license_plates_numbers = license_plate_number(license_plate_crop_thresh)[0]
license_plates_numbers = license_plate_number(license_plate_crop)[0]


numbers=[]
detections=[]
image = license_plate_crop
# print("hello=",license_plates_numbers)
for license_plate_number in license_plates_numbers.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate_number
            # sum=y1+y2
            # print("sum",sum)
            # print(license_plate_number)
            # print("class id",class_id)
            # numbers.append([class_id])
            detections.append([x1, y1, x2, y2, class_id])
            top_left = (x1, y1)
            bottom_right = (x2, y2)
            if class_id==0.0:
                object_name='0'
            elif class_id==1.0:
                object_name='1'
            elif class_id==2.0:
                object_name='2'
            elif class_id==3.0:
                object_name='3'
            elif class_id==4.0:
                object_name='4'
            elif class_id==5.0:
                object_name='5'
            elif class_id==6.0:
                object_name='6'
            elif class_id==7.0:
                object_name='7'
            elif class_id==8.0:
                object_name='8'
            elif class_id==9.0:
                object_name='9'
            elif class_id==10.0:
                object_name='Ba'
            elif class_id==11.0:
                object_name='Cha'
            elif class_id==12.0:
                object_name='Ga'
            elif class_id==13.0:
                object_name='Ja'
            elif class_id==14.0:
                object_name='Jha'
            elif class_id==15.0:
                object_name='Ka'
            elif class_id==16.0:
                object_name='Kha'
            elif class_id==17.0:
                object_name='Lu'
            elif class_id==18.0:
                object_name='Pa'
                    

            # Call the function to add class ID to the object's name
            add_class_id(image, class_id, object_name, top_left=top_left, bottom_right=bottom_right)
# vehicles = [2, 3, 5, 7]
# print("platenumber=",numbers)

#start test

#end test


image=cv2.resize(image,(500,500))
cv2.imshow("Image with Class ID", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Example list of detections (x1, y1, x2, y2, class_id)
# detections = [(10, 50, 30, 70, 1), (20, 30, 40, 50, 2), (5, 60, 25, 80, 3)]

# Convert the list of detections to a numpy array for easier manipulation
detections_array = np.array(detections)

# Sort the detections by y-coordinate (column index 1) and then x-coordinate (column index 0)
sorted_detections_array = detections_array[np.lexsort((detections_array[:, 0], detections_array[:, 1]))]

# Convert the sorted numpy array back to a list of detections
sorted_detections = sorted_detections_array.tolist()

# Print the sorted detections
for detection in sorted_detections:
    x1, y1, x2, y2, class_id = detection
    # if class_id==0.0:
    #     object_name='0'
    # elif class_id==1.0:
    #     object_name='1'
    # elif class_id==2.0:
    #     object_name='2'
    # elif class_id==3.0:
    #     object_name='3'
    # elif class_id==4.0:
    #     object_name='4'
    # elif class_id==5.0:
    #     object_name='5'
    # elif class_id==6.0:
    #     object_name='6'
    # elif class_id==7.0:
    #     object_name='7'
    # elif class_id==8.0:
    #     object_name='8'
    # elif class_id==9.0:
    #     object_name='9'
    # elif class_id==10.0:
    #     object_name='Ba'
    # elif class_id==11.0:
    #     object_name='Cha'
    # elif class_id==12.0:
    #     object_name='Ga'
    # elif class_id==13.0:
    #     object_name='Ja'
    # elif class_id==14.0:
    #     object_name='Jha'
    # elif class_id==15.0:
    #     object_name='Ka'
    # elif class_id==16.0:
    #     object_name='Kha'
    # elif class_id==17.0:
    #     object_name='Lu'
    # elif class_id==18.0:
    #     object_name='Pa'
    # numbers.append([object_name])
    
# print(numbers)
# print(sorted_detections)
# # result_string = "".join(numbers)
# print(result_string)
result_string = ''.join(''.join(row) for row in numbers)

# Print or use the result
print("y and then x=",result_string)
list1 = sorted_detections[:len(sorted_detections)//2]
list2 = sorted_detections[len(sorted_detections)//2:]

# Print the result
# print("List 1:", list1)
# print("List 2:", list2)

# detections = [(10, 50, 30, 70, 1), (20, 30, 40, 50, 2), (5, 60, 25, 80, 3)]

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
    if class_id==0.0:
        object_name='0'
    elif class_id==1.0:
        object_name='1'
    elif class_id==2.0:
        object_name='2'
    elif class_id==3.0:
        object_name='3'
    elif class_id==4.0:
        object_name='4'
    elif class_id==5.0:
        object_name='5'
    elif class_id==6.0:
        object_name='6'
    elif class_id==7.0:
        object_name='7'
    elif class_id==8.0:
        object_name='8'
    elif class_id==9.0:
        object_name='9'
    elif class_id==10.0:
        object_name='Ba'
    elif class_id==11.0:
        object_name='Cha'
    elif class_id==12.0:
        object_name='Ga'
    elif class_id==13.0:
        object_name='Ja'
    elif class_id==14.0:
        object_name='Jha'
    elif class_id==15.0:
        object_name='Ka'
    elif class_id==16.0:
        object_name='Kha'
    elif class_id==17.0:
        object_name='Lu'
    elif class_id==18.0:
        object_name='Pa'
    corrected_numbers.append([object_name])
    
# print(numbers)
# print(sorted_detections)
# result_string1= "".join(corrected_numbers)
# print(result_string)
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
    if class_id==0.0:
        object_name='0'
    elif class_id==1.0:
        object_name='1'
    elif class_id==2.0:
        object_name='2'
    elif class_id==3.0:
        object_name='3'
    elif class_id==4.0:
        object_name='4'
    elif class_id==5.0:
        object_name='5'
    elif class_id==6.0:
        object_name='6'
    elif class_id==7.0:
        object_name='7'
    elif class_id==8.0:
        object_name='8'
    elif class_id==9.0:
        object_name='9'
    elif class_id==10.0:
        object_name='Ba'
    elif class_id==11.0:
        object_name='Cha'
    elif class_id==12.0:
        object_name='Ga'
    elif class_id==13.0:
        object_name='Ja'
    elif class_id==14.0:
        object_name='Jha'
    elif class_id==15.0:
        object_name='Ka'
    elif class_id==16.0:
        object_name='Kha'
    elif class_id==17.0:
        object_name='Lu'
    elif class_id==18.0:
        object_name='Pa'
    corrected_numbers.append([object_name])
    
# print(numbers)
# print(sorted_detections)
# result_string2= "".join(corrected_numbers)
# print(result_string)
result_string= ''.join(''.join(row) for row in corrected_numbers)
print(result_string)



# # read frames
# frame_nmr = -1
# ret = True
# while ret:
#     frame_nmr += 1
#     ret, frame = cap.read()
#     if ret:
#         results[frame_nmr] = {}
#         # detect vehicles
#         detections = coco_model(frame)[0]
#         detections_ = []
#         for detection in detections.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = detection
#             if int(class_id) in vehicles:
#                 detections_.append([x1, y1, x2, y2, score])

#         # track vehicles
#         track_ids = mot_tracker.update(np.asarray(detections_))

#         # detect license plates
#         license_plates = license_plate_detector(frame)[0]
#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = license_plate

#             # assign license plate to car
#             xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

#             if car_id != -1:

#                 # crop license plate
#                 license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

#                 # process license plate
#                 license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#                 _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
#                 # cv2.imshow('window',license_plate_crop_thresh)

#                 # read license plate number
#                 license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

#                 if license_plate_text is not None:
#                     results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                                                   'license_plate': {'bbox': [x1, y1, x2, y2],
#                                                                     'text': license_plate_text,
#                                                                     'bbox_score': score,
#                                                                     'text_score': license_plate_text_score}}

# # write results
# write_csv(results, './test1.csv')