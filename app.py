from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import time
import os
import torch
import torchvision
from PIL import Image
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import json

app = Flask(__name__)

# Camera initialization
camera = None
backends = [cv2.CAP_V4L2, cv2.CAP_ANY, cv2.CAP_FFMPEG]
for backend in backends:
    for index in range(10):
        temp_camera = cv2.VideoCapture(index, backend)
        if temp_camera.isOpened():
            camera = temp_camera
            print(f"Camera opened at index {index} with backend {backend}")
            break
        temp_camera.release()
    if camera is not None:
        break

if camera is None:
    for index in range(10):
        pipeline = f"v4l2src device=/dev/video{index} ! videoconvert ! appsink"
        temp_camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if temp_camera.isOpened():
            camera = temp_camera
            print(f"Camera opened at index {index} with GStreamer")
            break
        temp_camera.release()

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trash info
TRASH_INFO_PATH = "trash_info.json"
try:
    with open(TRASH_INFO_PATH, "r", encoding="utf-8") as f:
        trash_info = json.load(f)
except Exception as e:
    print(f"Error loading trash_info.json: {e}")
    trash_info = {}

# Label mapping for Faster R-CNN
FASTER_RCNN_LABEL_MAP = {
    1: "can",
    2: "cigarette",
    3: "glass",
    4: "paper waste",
    5: "plastic bag",
    6: "plastic bottle"
}

# Models
models = {
    "yolov12": None,
    "detr": None,
    "fasterrcnn": None
}

# Initialize YOLOv12
try:
    models["yolov12"] = YOLO("model/best.pt")
except Exception as e:
    print(f"Error loading YOLOv12 model: {e}")

def load_detr_model():
    if models["detr"] is None:
        try:
            models["detr"] = torch.hub.load(
                'facebookresearch/detr:main',
                'detr_resnet50',
                pretrained=False,
                num_classes=6
            )
            state_dict = torch.load("model/Deformable-DETR-fine-tuned.pth", map_location="cpu")
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            missing, unexpected = models["detr"].load_state_dict(state_dict, strict=False)
            print("DETR missing keys:", missing, "unexpected keys:", unexpected)
            models["detr"].eval()
        except Exception as e:
            print(f"Error loading DETR model: {e}")
            models["detr"] = None
    return models["detr"]

def load_fasterrcnn_model():
    if models["fasterrcnn"] is None:
        try:
            model = fasterrcnn_resnet50_fpn(weights=None, num_classes=7)
            state_dict = torch.load("model/fasterrcnn_resnet50.pth", map_location="cpu")
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("Faster R-CNN missing keys:", missing, "unexpected keys:", unexpected)
            model.eval()
            models["fasterrcnn"] = model
        except Exception as e:
            print(f"Error loading Faster R-CNN model: {e}")
            models["fasterrcnn"] = None
    return models["fasterrcnn"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_and_draw(frame):
    try:
        h, w, _ = frame.shape
        roi_w, roi_h = int(w * 0.7), int(h * 0.7)
        left, top = (w - roi_w)//2, (h - roi_h)//2
        right, bottom = left + roi_w, top + roi_h
        roi = frame[top:bottom, left:right]
        if models["yolov12"] is None:
            raise Exception("YOLOv12 model not loaded")
        results = models["yolov12"](roi)
        annotated_roi = results[0].plot()
        frame_copy = frame.copy()
        frame_copy[top:bottom, left:right] = annotated_roi
        return frame_copy, results
    except Exception as e:
        print(f"Error in classify_and_draw: {e}")
        raise

def generate_frames():
    if camera is None:
        print("No camera available for video feed")
        return
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera")
            break
        try:
            annotated_frame, _ = classify_and_draw(frame)
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print("Failed to encode frame")
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            break

def normalize_label(name):
    if name is None:
        return None
    return name.strip().lower().replace("_", " ")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if camera is None:
        return jsonify({"error": "No camera available"}), 500
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/classify', methods=['GET'])
def classify():
    try:
        if camera is None:
            return jsonify({"error": "No camera available"}), 500
        success, frame = camera.read()
        if not success:
            return jsonify({"error": "No frame"}), 500
        _, results = classify_and_draw(frame)
        print("Boxes detected:", len(results[0].boxes) if results[0].boxes else 0)
        trash_type = results[0].names[results[0].boxes.cls[0].item()] if results[0].boxes else None
        if trash_type:
            mode = "pickup_and_drop"
            bin_coords = {
                "can": [-0.06, 0.42], "cigarette": [0.08, 0.42], "glass": [0.08, 0.28],
                "paper waste": [0.08, 0.42], "plastic bag": [-0.06, 0.28], "plastic bottle": [-0.06, 0.28]
            }
            coords = {"drop": bin_coords.get(trash_type, [0, 0])}
        else:
            mode, coords = "idle", {}
        return jsonify({"trash_type": trash_type, "mode": mode, "coords": coords})
    except Exception as e:
        print(f"Error in classify: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files or not request.files['file'] or request.files['file'].filename == '':
            return redirect(request.url)
        file = request.files['file']
        model_name = request.form.get('model', 'yolov12')
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            image = Image.open(filepath).convert("RGB")
        except Exception as e:
            print(f"Error opening image: {e}")
            return jsonify({"error": "Invalid image file"}), 400

        img_np = np.array(image)
        annotated_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        detections = []
        color_bgr = (255, 0, 0)  # Blue in BGR

        if model_name == "yolov12":
            if models["yolov12"] is None:
                return jsonify({"error": "YOLOv12 model not loaded"}), 500
            results = models["yolov12"](img_np)
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes else []
            confs = r.boxes.conf.cpu().numpy() if r.boxes else []
            cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes else []
            names = r.names
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                label = normalize_label(names.get(cls_id, str(cls_id)))
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(annotated_bgr, f"{label} {conf:.2f}", (x1, max(0, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                detections.append({
                    "label": label, "confidence": float(conf), "bbox": [x1, y1, x2, y2],
                    "info": trash_info.get(label, {})
                })

        elif model_name == "detr":
            model = load_detr_model()
            if model is None:
                return jsonify({"error": "DETR model not loaded"}), 500
            preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize(800),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.7
            if not keep.any():
                keep = probas.max(-1).values > 0.5
            boxes = outputs['pred_boxes'][0, keep].cpu().numpy()
            scores = probas[keep].max(-1).values.cpu().numpy()
            labels = probas[keep].argmax(-1).cpu().numpy()
            h, w = img_np.shape[:2]
            for box, score, lbl in zip(boxes, scores, labels):
                cx, cy, bw, bh = box
                x1, y1, x2, y2 = int((cx - bw/2) * w), int((cy - bh/2) * h), int((cx + bw/2) * w), int((cy + bh/2) * h)
                label = normalize_label(f"class_{lbl}")
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(annotated_bgr, f"{label} {score:.2f}", (x1, max(0, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                detections.append({
                    "label": label, "confidence": float(score), "bbox": [x1, y1, x2, y2],
                    "info": trash_info.get(label, {})
                })

        elif model_name == "fasterrcnn":
            model = load_fasterrcnn_model()
            if model is None:
                return jsonify({"error": "Faster R-CNN model not loaded"}), 500
            preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            img_tensor = preprocess(image)
            with torch.no_grad():
                outputs = model([img_tensor])
            scores = outputs[0]['scores'].cpu().numpy()
            labels = outputs[0]['labels'].cpu().numpy()
            boxes = outputs[0]['boxes'].cpu().numpy()
            keep = scores > 0.5
            for box, score, lbl in zip(boxes[keep], scores[keep], labels[keep]):
                x1, y1, x2, y2 = map(int, box)
                label = normalize_label(FASTER_RCNN_LABEL_MAP.get(int(lbl), f"class_{lbl}"))
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(annotated_bgr, f"{label} {score:.2f}", (x1, max(0, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
                detections.append({
                    "label": label, "confidence": float(score), "bbox": [x1, y1, x2, y2],
                    "info": trash_info.get(label, {})
                })

        else:
            return jsonify({"error": f"Unknown model: {model_name}"}), 400

        annotated_name = f"annotated_{filename}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_name)
        cv2.imwrite(annotated_path, annotated_bgr)
        return render_template(
            'result.html',
            model_name=model_name,
            original_image=filepath,
            annotated_image=annotated_path,
            detections=detections
        )

    except Exception as e:
        print(f"Error in detect: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/check_camera', methods=['GET'])
def check_camera():
    return jsonify({"camera_available": camera is not None})

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        if camera is not None:
            camera.release()
            print("Camera released")