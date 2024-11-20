from ultralytics import YOLO
import cv2
import torch

# Load the YOLOv8 model with custom weights
model = YOLO('non_augmented_weights.pt')

# Start video capture (use 0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize the frame (optional, adjust as needed)
    # frame_resized = cv2.resize(frame, (640, 640))

    # Perform inference using YOLOv8 model
    results = model(frame)

    # Extract the bounding boxes and class labels
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().numpy()

        if conf > 0.5:  # Only display boxes with high confidence
            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Optionally, put a label with the confidence
            cv2.putText(frame, f'Bottle {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the frame with detections
    cv2.imshow('Real-Time Bottle Detection', frame)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
