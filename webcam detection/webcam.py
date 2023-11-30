# Import necessary libraries
from ultralytics import YOLO  # Import the YOLO object detection library
# Load the pre-trained YOLOv8 model from the specified file path
model = YOLO("best_N.pt")
# Perform object detection on the video stream from the specified source (in this case, webcam 0)
# True to display the detected objects on the video stream
results = model.predict(source="0", show=True)
# Print the results of the object detection
print(results)
