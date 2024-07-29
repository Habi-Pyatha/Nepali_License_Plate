import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_nepali_best.pt')
license_plate_number = YOLO('./models/license_plate_number_nepali_best.pt')

def get_object_name(class_id):
    # Mapping class IDs to object names
    class_names = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'Ba', 11: 'Cha', 12: 'Ga', 13: 'Ja', 14: 'Jha', 15: 'Ka', 16: 'Kha', 17: 'Lu', 18: 'Pa'
    }
    return class_names.get(int(class_id), 'Unknown')

def add_class_id(image, class_id, object_name, top_left, bottom_right):
    # Function to draw bounding boxes and labels
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
    text = f"{object_name}"

    # Calculate the position for the text
    text_position = (top_left[0] + 50, top_left[1] + 20)

    # Draw the text on the image
    cv2.putText(image, text, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

def process_image(image_path):
    # Load the selected image
    frame = cv2.imread(image_path)

    # Perform license plate detection
    license_plates = license_plate_detector(frame)[0]

    # Process the detected license plates
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Crop the detected license plate
        license_plate_crop = frame[int(y1 - 10):int(y2), int(x1): int(x2), :]
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

        # Perform further processing if needed
        # ...

        # Display cropped license plate
        cv2.imshow('Cropped License Plate', cv2.resize(license_plate_crop, (500, 500)))

        # Display grayscale version
        cv2.imshow('Grayscale', license_plate_crop_gray)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        image_adapteq = clahe.apply(license_plate_crop_gray)
        cv2.imshow('Equalized', image_adapteq)

        # Perform license plate number detection
        license_plates_numbers = license_plate_number(license_plate_crop)[0]

        # Process and display the detected numbers
        for license_plate_number in license_plates_numbers.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate_number
            top_left = (x1, y1)
            bottom_right = (x2, y2)
            object_name = get_object_name(class_id)
            add_class_id(license_plate_crop, class_id, object_name, top_left=top_left, bottom_right=bottom_right)

        cv2.imshow('Final Result', cv2.resize(license_plate_crop, (500, 500)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Create a Tkinter root window
root = tk.Tk()
root.title("License Plate Recognition")

# Function to handle button click event
def open_image():
    # Ask the user to choose an image file using a file dialog
    image_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    # Process the selected image
    if image_path:
        process_image(image_path)

# Create left and right frames
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Create and pack a button to open image
open_button = tk.Button(left_frame, text="Open Image", command=open_image)
open_button.pack(pady=10)

# Create labels to display results
result_label = tk.Label(left_frame, text="Results will be shown here", wraplength=300)
result_label.pack(pady=10)

# Create a canvas to display the image
canvas = tk.Canvas(right_frame, width=500, height=500)
canvas.pack()

# Function to display image on canvas
def display_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (500, 500))
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.image = img  # Keep a reference to the image to prevent garbage collection

# Start the Tkinter event loop
root.mainloop()
