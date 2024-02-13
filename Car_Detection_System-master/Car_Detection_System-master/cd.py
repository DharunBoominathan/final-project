import cv2
import numpy as np

# Load pre-trained YOLOv3 model for car detection
net = cv2.dnn.readNet("yolov3.weigths","yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Function to perform car detection
def detect_cars(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(layer_names)

    bounding_boxes = []
    confidences = []
    
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 2:  # Class ID 2 corresponds to cars in COCO dataset
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                bounding_boxes.append((x, y, x + w, y + h))
                confidences.append(float(confidence))

    return bounding_boxes, confidences

# Open the video file
cap = cv2.VideoCapture("vb2.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform car detection
    bounding_boxes, confidences = detect_cars(frame)

    for box in zip(bounding_boxes, confidences):
        x, y, x2, y2 = box[0]
        confidence = box[1]

        # Draw bounding box on the frame
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Car Detection", frame)
    

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
