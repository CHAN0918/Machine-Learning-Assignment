import torch
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use a larger model for better accuracy
#model = YOLO("D:\\Users\\admin\\Documents\\runs\\stationery_model\\weights\\best.pt")  # Update with the path to your trained model weights
# Open the webcam
cap = cv2.VideoCapture(0)


# Load the trained YOLO model
# model = YOLO("C:\\Users\\PC\\yolov5\\runs\\detect\\train3\\weights\\best.pt")  # Update with the path to your trained model weights

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set a confidence threshold
confidence_threshold = 0.6  # Adjust this value as needed


# Set frame processing interval
frame_interval = 100  # Process every 2nd frame
frame_count = 0

# Print out the class names
class_names = model.names
print("Classes used by the model:", class_names)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Process every nth frame to improve performance
    if frame_count % frame_interval == 0:
        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (320, 240))
    # Perform object detection
    results = model(frame)

    # Extract bounding boxes and labels from the results
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # coordinates of the bounding box
                conf = box.conf.item()  # confidence score (convert to a Python float)
                cls = box.cls.item()  # class id (convert to a Python float)
                label = model.names[int(cls)]
                if conf > confidence_threshold:  # Filter by confidence
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv8 Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
