# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:26:55 2023

@author: Dell
"""

# from ultralytics import YOLO
# # from yolo import yolo
# # from ultralytics.yolo.v8.detect.predict import DetectionPredictor
# import cv2
# model=YOLO("C:/Users/Dell/pythons/train-yolov8-custom-dataset-step-by-step-guide-master/local_env/runs/detect/train3/weights/best.pt")
# #model = YOLO("C:\yolov8\runs\detect\train5\weights\best.pt")
# #model = YOLO(/home/esp/yolov8/best.pt)
# model.predict(source='0',show=True ,conf=0.5)
# # results = model.predict(source="1", show=True)

# # print(*results)


import cv2
from ultralytics.yolo.v8 import YOLO

# Load the YOLO model
yolo = YOLO("C:/Users/Dell/pythons/train-yolov8-custom-dataset-step-by-step-guide-master/local_env/config.yaml", "C:/Users/Dell/pythons/train-yolov8-custom-dataset-step-by-step-guide-master/local_env/runs/detect/train3/weights/best.pt")

# Open a video capture stream
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify a file path for a video file

while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    if not ret:
        break

    # Perform object detection on the frame
    results = yolo(frame)

    # Render the detected objects on the frame
    frame_with_objects = results.render()[0]

    # Display the frame with detected objects
    cv2.imshow('YOLO Object Detection', frame_with_objects)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()

# Close the YOLO model when done
yolo.close()