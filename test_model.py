from ultralytics import YOLO
import cv2

model = YOLO('models/best.pt')

print("CLASSES:", model.names)

img = cv2.imread('https://media.istockphoto.com/id/627337336/photo/cars-trucks-and-rescue-vehicle-driving-in-dangerous-winter-weather.jpg?s=612x612&w=0&k=20&c=FrSLP1G3ZYfiOvRf5hKbA9Fju5kpQ7KdI7Gw850anYI=')

if img is None:
    print("❌ Image not found. Put a test image in static/uploads/")
else:
    results = model(img, conf=0.6)
    results[0].show()