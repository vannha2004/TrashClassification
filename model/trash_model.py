# model/trash_model.py
from ultralytics import YOLO
import cv2

# Load model YOLO
model = YOLO("model/best.pt")

def classify_and_draw(frame):
    """
    Nhận diện rác trong vùng giữa khung hình.
    Trả về frame đã vẽ bounding box.
    """
    h, w, _ = frame.shape

    # Xác định ROI giữa khung hình (50% kích thước gốc)
    roi_w, roi_h = int(w * 0.5), int(h * 0.5)
    left = (w - roi_w) // 2
    top = (h - roi_h) // 2
    right = left + roi_w
    bottom = top + roi_h

    # Cắt ROI
    roi = frame[top:bottom, left:right]

    # Nhận diện chỉ trong ROI
    results = model(roi)

    # Lấy ảnh có bounding box từ YOLO
    annotated_roi = results[0].plot()

    # Gắn ROI đã annotate vào lại frame gốc
    frame_copy = frame.copy()
    frame_copy[top:bottom, left:right] = annotated_roi

    return frame_copy
