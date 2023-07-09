import cv2
import numpy as np

# Load the YOLOv2 model
net = cv2.dnn.readNetFromDarknet(r'D:\projects\Python-Computer Vision\computer vision tutorial\krish naik course\yolo car detection\YAD2K-master\YAD2K-master\yolov2.cfg', r'D:\projects\Python-Computer Vision\computer vision tutorial\krish naik course\yolo car detection\YAD2K-master\YAD2K-master\yolov2.weights')

# Load the COCO class labels
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Load the video
video_path = r'D:\projects\Python-Computer Vision\computer vision tutorial\krish naik course\yolo car detection\cars_on_highway (1080p).mp4'
cap = cv2.VideoCapture(video_path)

# Initialize a counter to keep track of the car count
car_count = 0

while True:
    # Read each frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform blob conversion for input to the neural network
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    
    # Set the input blob for the neural network
    net.setInput(blob)
    
    # Run the forward pass to get the detections
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    layer_outputs = net.forward(output_layers)
    
    # Initialize lists for bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    
    # Process each output layer
    for output in layer_outputs:
        for detection in output:
            # Retrieve class probabilities and bounding box coordinates
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and class_id == 2:  # Class ID 2 represents cars in COCO dataset
                # Scale the bounding box coordinates to the original frame size
                width, height = frame.shape[1], frame.shape[0]
                center_x, center_y, w, h = detection[:4] * np.array([width, height, width, height])
                x, y = int(center_x - w/2), int(center_y - h/2)
                
                # Add the bounding box, confidence, and class ID to the respective lists
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw bounding boxes and count the cars
    for i in indices:
        i = 0
        x, y, w, h = boxes[i]
        
        # Draw the bounding box rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Increment the car count
        car_count += 1
    
    # Display the frame with bounding boxes
    cv2.imshow('Car Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Print the total car countq
print("Total cars detected:", car_count)
