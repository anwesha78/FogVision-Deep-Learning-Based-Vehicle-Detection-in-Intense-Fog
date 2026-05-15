import cv2
from ultralytics import YOLO
from dehaze import dehaze_image

# Load YOLO model
model = YOLO("yolov8n.pt")

# Image path
image_path = "../dataset/foggy/test.jpg"

# Read image
image = cv2.imread(image_path)

# Dehaze the image
dehazed = dehaze_image(image)

# Run detection on dehazed image
results = model(dehazed)

# Draw bounding boxes
detected = results[0].plot()
# Save output images
cv2.imwrite("../outputs/original.jpg", image)
cv2.imwrite("../outputs/dehazed.jpg", dehazed)
cv2.imwrite("../outputs/detected.jpg", detected)

# Show outputs
cv2.imshow("Original Foggy Image", image)
cv2.imshow("Dehazed Image", dehazed)
cv2.imshow("Detection After Dehazing", detected)

cv2.waitKey(0)
cv2.destroyAllWindows()