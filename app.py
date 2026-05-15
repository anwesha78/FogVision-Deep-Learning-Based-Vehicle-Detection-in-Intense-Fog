from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import cv2
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#  LOAD MODEL
MODEL_PATH = 'models/best.pt'
model = YOLO(MODEL_PATH)

print("\n🔥 MODEL LOADED 🔥")
print("Path:", MODEL_PATH)
print("Classes:", model.names)
print("🔥 READY 🔥\n")


#  VERY LIGHT DEHAZE (SAFE)
def slight_dehaze(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    v = clahe.apply(v)

    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@app.route('/', methods=['GET', 'POST'])
def index():
    before_image = after_image = None
    before_video = after_video = None
    error = None

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            error = "No file selected"
        else:
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            input_path = os.path.join(UPLOAD_FOLDER, unique_name)

            file.save(input_path)
            print("➡️ New file received:", input_path)

            ext = filename.split('.')[-1].lower()

            # ================= IMAGE =================
            if ext in ['jpg', 'jpeg', 'png']:

                img = cv2.imread(input_path)

                # DETECTION ON ORIGINAL IMAGE
                results = model(img, conf=0.6)

                detected = [model.names[int(c)] for c in results[0].boxes.cls]
                print("➡️ Detected:", detected)

                annotated = results[0].plot()

                #  APPLY DEHAZE AFTER DETECTION
                enhanced = slight_dehaze(img)

                # BLEND (CLEAN PROFESSIONAL OUTPUT)
                final = cv2.addWeighted(enhanced, 0.6, annotated, 0.4, 0)

                output_name = f"{uuid.uuid4().hex}.jpg"
                output_path = os.path.join(OUTPUT_FOLDER, output_name)

                cv2.imwrite(output_path, final)
                print("➡️ Output saved:", output_path)

                before_image = unique_name
                after_image = output_name

            # ================= VIDEO =================
            elif ext in ['mp4', 'avi', 'mov']:

                cap = cv2.VideoCapture(input_path)

                width = int(cap.get(3))
                height = int(cap.get(4))
                fps = cap.get(5)

                if fps <= 0:
                    fps = 25

                output_name = f"{uuid.uuid4().hex}.mp4"
                output_path = os.path.join(OUTPUT_FOLDER, output_name)

                out = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (width, height)
                )

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    #  DETECTION ON ORIGINAL FRAME
                    results = model(frame, conf=0.6)
                    annotated = results[0].plot()

                    # DEHAZE AFTER
                    enhanced = slight_dehaze(frame)

                    final = cv2.addWeighted(enhanced, 0.6, annotated, 0.4, 0)

                    out.write(final)

                cap.release()
                out.release()

                before_video = unique_name
                after_video = output_name

            else:
                error = "Unsupported file type"

    return render_template(
        'index.html',
        before_image=before_image,
        after_image=after_image,
        before_video=before_video,
        after_video=after_video,
        error=error
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5500, debug=False, use_reloader=False)