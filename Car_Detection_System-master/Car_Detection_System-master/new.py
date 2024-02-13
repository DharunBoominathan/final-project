import cv2
import numpy as np
import os

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Error handling for network loading
if net.empty():
    print("Error: Failed to load network.")
    exit()

# Get output layer names
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indices]

# Load video or images
video = cv2.VideoCapture("vb.mp4")

# Error handling for video loading
if not video.isOpened():
    print("Error: Failed to open video.")
    exit()

# Directory to save detected car images
output_dir = "detected_cars"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Dictionary to store unique car images
unique_cars = {}

# Loop through video frames
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    height, width, channels = frame.shape
    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process each detected object
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2: # 2 is the class ID for car in COCO dataset
                # Object detected is a car with confidence greater than 50%
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Extract the detected car image
                car_image = frame[y:y+h, x:x+w]
                
                # Generate a unique identifier for the car based on its bounding box coordinates
                car_id = (x, y, x+w, y+h)
                
                # Check if the car is already detected
                if car_id not in unique_cars:
                    # If not, add it to the dictionary
                    unique_cars[car_id] = car_image
                    
                    # Save the detected car image
                    cv2.imwrite(os.path.join(output_dir, f"car_{len(unique_cars)}.jpg"), car_image)
                    
                    # Display the detected car image
                    cv2.imshow("Detected Car", car_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

# Now you have a dictionary 'unique_cars' containing unique car images
# You can further process these images to filter out the best one based on your criteria
