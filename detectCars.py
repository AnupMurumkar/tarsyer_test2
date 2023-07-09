import cv2
import numpy as np
from centroidtracker import CentroidTracker

# Load YOLOv3 weights and configuration
net = cv2.dnn.readNetFromDarknet(r'D:\projects\Python-Computer Vision\computer vision tutorial\krish naik course\yolo car detection\YAD2K-master\YAD2K-master\yolov2.cfg', r'D:\projects\Python-Computer Vision\computer vision tutorial\krish naik course\yolo car detection\YAD2K-master\YAD2K-master\yolov2.weights')
 

# Load COCO class labels
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Set the minimum confidence threshold for detections
conf_threshold = 0.6

# Set the non-maximum suppression threshold
nms_threshold = 0.4

# Load the video
video_path = r'D:\projects\Python-Computer Vision\computer vision tutorial\krish naik course\yolo car detection\cars_on_highway (360p).mp4'

cap = cv2.VideoCapture(video_path)

# Create a CentroidTracker object
centroid_tracker = CentroidTracker()

while True:
    # Read each frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Create a blob from the input frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Run the forward pass to get the output layer names and predictions
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    # Initialize lists for bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Process each output layer
    for output in layer_outputs:
        # Process each detection
        for detection in output:
            # Get the class probabilities and the index of the highest score
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections below the confidence threshold
            if confidence > conf_threshold and class_id == 2:  # Class ID 2 represents cars in COCO dataset
                # Scale the bounding box coordinates to the frame size
                width, height = frame.shape[1], frame.shape[0]
                center_x, center_y, w, h = detection[:4] * np.array([width, height, width, height])
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                # Add the bounding box, confidence, and class ID to the respective lists
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Get the list of bounding box rectangles
    rects = []
    for i in indices:
        x, y, w, h = boxes[i]
        rects.append((x, y, x + w, y + h))

    # Update the centroid tracker
    objects = centroid_tracker.update(rects)

    # Draw bounding boxes and labels on the frame
    for (object_id, centroid) in objects.items():
        x, y = centroid
        label = f"Car {object_id}"

    # Draw the bounding box rectangle and label
    cv2.rectangle(frame, (x, y), (x + 1, y + 1), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes and labels
    cv2.imshow('Car Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
