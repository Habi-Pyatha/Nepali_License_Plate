from ultralytics import YOLO
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import numpy as np

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')
license_plate_number = YOLO('./models/license_plate_number_nepali_best.pt')
# load video
cap = cv2.VideoCapture('./test.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                # license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                # cv2.imshow('window',license_plate_crop_thresh)
                # license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)
                
                # read license plate number
                # license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                # frame=cv2.imread("./eight.jpg")
                # img=cv2.resize(frame,(500,500))
                # cv2.imshow('img',img)
                # cv2.waitKey(0)
                # license_plates = license_plate_detector(frame)[0]
                # for license_plate in license_plates.boxes.data.tolist():
                #             x1, y1, x2, y2, score, class_id = license_plate
                #             # print(license_plate)
                # license_plate_crop = frame[int(y1-10):int(y2), int(x1): int(x2), :]
                # # license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                # # license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                # img=cv2.resize(license_plate_crop,(500,500))
                # cv2.imshow('img',img)
                # cv2.waitKey(0)
                license_plates_numbers_list = license_plate_number(license_plate_crop)[0]
                numbers=[]
                
                detections=[]
                
                for license_plate_num in license_plates_numbers_list.boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = license_plate_num
                           
                            detections.append([x1, y1, x2, y2, class_id])
                            
            
                
            
               
            
                # Convert the list of detections to a numpy array for easier manipulation
                detections_array = np.array(detections)
                if detections_array.ndim == 1:
                    # 1-dimensional array, directly use np.argsort
                    sorted_detections_array = detections_array[np.argsort(detections_array)]
                else:
                    # 2-dimensional array, sort by x-coordinate (column index 0)
                    sorted_detections_array = detections_array[np.lexsort((detections_array[:, 0], detections_array[:, 1]))]
                # Sort the detections by y-coordinate (column index 1) and then x-coordinate (column index 0)
                # sorted_detections_array = detections_array[np.lexsort((detections_array[:, 0], detections_array[:, 1]))]
            
                # Convert the sorted numpy array back to a list of detections
                sorted_detections = sorted_detections_array.tolist()
            
                # Print the sorted detections
                for detection in sorted_detections:
                    x1, y1, x2, y2, class_id = detection
                    
                result_string = ''.join(''.join(row) for row in numbers)
            
                # Print or use the result
                print(result_string)
                list1 = sorted_detections[:len(sorted_detections)//2]
                list2 = sorted_detections[len(sorted_detections)//2:]
            
                
                detections_array = np.array(list1)
                # detections_array = np.array(detections)

                # Check the shape of detections_array
                if detections_array.ndim == 1:
                    # 1-dimensional array, directly use np.argsort
                    sorted_detections_array = detections_array[np.argsort(detections_array)]
                else:
                    # 2-dimensional array, sort by x-coordinate (column index 0)
                    sorted_detections_array = detections_array[np.argsort(detections_array[:, 0])]
                
                # # Convert the sorted numpy array back to a list of detections
                # sorted_detections = sorted_detections_array.tolist()
            
                # # Sort the detections by x-coordinate (column index 0) only
                # sorted_detections_array = detections_array[np.argsort(detections_array[:, 0])]
                
            
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
                    
               
                result_string= ''.join(''.join(row) for row in corrected_numbers)
                # print(result_string)
                    
                detections_array = np.array(list2)
                if detections_array.ndim == 1:
                    # 1-dimensional array, directly use np.argsort
                    sorted_detections_array = detections_array[np.argsort(detections_array)]
                else:
                    # 2-dimensional array, sort by x-coordinate (column index 0)
                    sorted_detections_array = detections_array[np.argsort(detections_array[:, 0])]
                # Sort the detections by x-coordinate (column index 0) only
                # sorted_detections_array = detections_array[np.argsort(detections_array[:, 0])]
            
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
                    
                
                result_string= ''.join(''.join(row) for row in corrected_numbers)
                # print(result_string)
                license_plate_text, license_plate_text_score=result_string,90.0
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
# def license_plate_number(image):
#     # Your function implementation here
#     # ...
#     return result_list
# write results
write_csv(results, './test.csv')