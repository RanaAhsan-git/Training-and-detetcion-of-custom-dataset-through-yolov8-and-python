import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('E:\weights\Ahsan pen.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Make predictions
    results = model(frame)

    # Visualize the results
    annotated_frame = results[0].plot()

    # Display the frame with bounding boxes
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
