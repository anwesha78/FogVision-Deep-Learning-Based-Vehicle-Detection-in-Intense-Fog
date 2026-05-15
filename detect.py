from ultralytics import YOLO

model = YOLO("integrated.pt")
results = model("static/test.jpg", save=True)

print("Detection complete")