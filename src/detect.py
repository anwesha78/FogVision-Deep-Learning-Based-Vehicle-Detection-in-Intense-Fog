from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Correct path
image_path = "../dataset/foggy/test.jpg"

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Image not found. Check the path.")
    exit()

results = model(image)

annotated = results[0].plot()

cv2.imshow("Detection Result", annotated)

cv2.waitKey(0)
cv2.destroyAllWindows()