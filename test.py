from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("model/best.pt")

# Print list of labels
print("Labels:", model.names)

# Print number of detectable trash types
print("Number of detectable trash types:", len(model.names))