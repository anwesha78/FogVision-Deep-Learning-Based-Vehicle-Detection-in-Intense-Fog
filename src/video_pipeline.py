import cv2
import numpy as np
from ultralytics import YOLO
from dehaze import dehaze_image

# Load YOLO model
model = YOLO("yolov8n.pt")

video_path = "../videos/foggy_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
out = cv2.VideoWriter(
    "../outputs/dehazed_detected_video.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width*2, frame_height)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Dehaze frame
    dehazed = dehaze_image(frame)

    # Run YOLO detection
    results = model(dehazed)

    detected = results[0].plot()

    # Combine original and processed frames side-by-side
    combined = np.hstack((frame, detected))

    # Save combined frame
    out.write(combined)

    # Display combined video
    cv2.imshow("Original vs Dehazed + Detection", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved in outputs folder!")