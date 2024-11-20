import cv2
import torch
import numpy as np
from ultralytics import YOLO  # Import the YOLO class directly

# Path to the YOLOv5 weights and image file
weights_path = "non_augmented_weights.pt"  # Replace with the path to your weights file
input_image = "bottles-1554.jpg"  # Replace with the path to your image file

# Load the YOLOv5 model directly from ultralytics package
model = YOLO(weights_path)  # Using the YOLO class to load the model

# Load input image
img = cv2.imread(input_image)

# Run inference (detection)
results = model(img)  # Perform detection

# `results` will contain the predictions (bounding boxes, class IDs, and confidence scores)
predictions = results[0].boxes.xyxy  # Get bounding box coordinates (x1, y1, x2, y2)
confidences = results[0].boxes.conf  # Get confidence scores
class_ids = results[0].boxes.cls  # Get class IDs

# Since you only have one class (bottle), you can label it directly
class_name = "bottle"  # Name of the class

# Draw bounding boxes and labels on the image
for i in range(len(predictions)):
    x1, y1, x2, y2 = predictions[i]
    confidence = confidences[i]

    # Draw the bounding box and label on the image
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box
    cv2.putText(img, f'{class_name} {confidence:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Save the output image with bounding boxes
cv2.imwrite("output.jpg", img)
print("Detection completed. Image saved as 'output.jpg'")
