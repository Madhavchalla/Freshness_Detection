import os
import sys
import base64
import numpy as np
import cv2
import torch
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define Base Directory for absolute path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------------
# PyTorch 2.6+ Compatibility Patch for YOLOv5 Weights Unpickling
# Prevents "Weights only load failed" / Gunicorn Exit Code 3 on Render
# -------------------------------------------------------------------
try:
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except Exception as patch_err:
    print(f"Warning patching torch.load: {patch_err}")

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load models at startup with absolute paths
model_path = os.path.join(BASE_DIR, "healthy_vs_rotten.h5")
yolo_weights_path = os.path.join(BASE_DIR, "yolov5s.pt")
yolo_local_dir = os.path.join(BASE_DIR, "yolov5")

print(f"Loading classification model from {model_path}...")
model = load_model(model_path)

print(f"Loading YOLOv5 detection model from {yolo_weights_path}...")
if os.path.exists(yolo_local_dir):
    # Load from local offline YOLOv5 directory
    yolo_model = torch.hub.load(yolo_local_dir, 'custom', path=yolo_weights_path, source='local')
else:
    # Fallback to online ultralytics hub
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights_path)

# Configure sensitivity for produce proposals (including decayed/rotten fruits)
yolo_model.conf = 0.10  # Detection confidence threshold
yolo_model.iou = 0.45   # IoU threshold for overlapping items

class_names = ['Fresh', 'Rotten']

# COCO IDs assigned to fruits, produce, and decayed round food objects
PRODUCE_COCO_IDS = {32, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55}

# COCO non-fruit background & container class IDs to explicitly filter out
NON_FRUIT_COCO_IDS = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 45, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79
}

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def apply_nms(candidates, iou_thresh=0.35):
    if not candidates:
        return []

    scored_candidates = []
    for box, score, src in candidates:
        # Boost YOLO produce detections slightly over raw contours
        effective_score = score + 0.25 if src == 'yolo_produce' else score
        scored_candidates.append((box, effective_score, src))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    keep = []
    while len(scored_candidates) > 0:
        current_box, current_score, current_src = scored_candidates.pop(0)
        keep.append(current_box)

        remaining = []
        for box, score, src in scored_candidates:
            if compute_iou(current_box, box) < iou_thresh:
                remaining.append((box, score, src))
        scored_candidates = remaining

    return keep

def extract_fruit_bounding_boxes(img):
    """
    Extracts bounding boxes for produce items (fresh or rotten).
    Excludes non-fruit objects like bowls, cups, tables, cutlery, and background clutter.
    Captures dark shriveled rotten fruits, un-boxed produce, and multi-fruit clusters cleanly.
    """
    img_h, img_w, _ = img.shape
    total_area = img_h * img_w
    candidate_boxes = []

    # 1. YOLOv5 Object Detection (Sensitivity 0.10)
    results = yolo_model(img)
    detections = results.xyxy[0]  # Tensor of detected bounding boxes

    if len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_id = int(cls.item())
            
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h

            # Reject full-frame background boxes (>85% of total image area)
            if box_area > (total_area * 0.85):
                continue
            # Reject extreme wall/table background strip boxes
            if box_w > (img_w * 0.90) and box_h > (img_h * 0.90):
                continue

            # Check class ID: If produce/fruit/decayed food candidate class
            if cls_id in PRODUCE_COCO_IDS:
                candidate_boxes.append(([x1, y1, x2, y2], float(conf.item()), 'yolo_produce'))
            elif cls_id not in NON_FRUIT_COCO_IDS:
                # General candidate object box (not an explicit bowl/table/cup)
                candidate_boxes.append(([x1, y1, x2, y2], float(conf.item()), 'yolo_generic'))

    # 2. Multi-Spectral OpenCV Saliency & Edge Blob Detector (Catches discolored/rotten fruits)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    saliency = cv2.addWeighted(sat, 0.5, mag, 0.5, 0)
    _, thresh = cv2.threshold(saliency, 25, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = total_area * 0.008  # 0.8% of image
    max_area = total_area * 0.70   # 70% of image

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = float(bw) / float(bh)
            if 0.35 <= aspect <= 2.8 and bw < (img_w * 0.85) and bh < (img_h * 0.85):
                candidate_boxes.append(([bx, by, bx + bw, by + bh], 0.50, 'opencv_contour'))

    # 3. Apply Non-Maximum Suppression to deduplicate overlapping boxes
    final_boxes = apply_nms(candidate_boxes, iou_thresh=0.35)
    return final_boxes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    wants_json = request.is_json or ('application/json' in request.headers.get('Accept', ''))

    if 'image' not in request.files:
        if wants_json:
            return jsonify({'error': 'No image file found'}), 400
        return 'No image file found'

    file = request.files['image']
    if file.filename == '':
        if wants_json:
            return jsonify({'error': 'No file selected'}), 400
        return 'No file selected'

    # Read image directly from RAM buffer into OpenCV
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        if wants_json:
            return jsonify({'error': 'Invalid image file'}), 400
        return 'Invalid image file'

    img_h, img_w, _ = img.shape
    annotated_img = img.copy()
    detected_items = []

    # Extract clean produce bounding boxes (filtering out bowls, cups, tables, top wall strips)
    fruit_boxes = extract_fruit_bounding_boxes(img)

    # Process localized produce boxes
    if len(fruit_boxes) > 0:
        for box in fruit_boxes:
            x1, y1, x2, y2 = box
            
            # Crop localized produce in RAM
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or (x2 - x1) < 15 or (y2 - y1) < 15:
                continue

            resized = cv2.resize(crop, (224, 224))
            img_array = img_to_array(resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict Fresh/Rotten for this specific fruit crop
            prediction = model.predict(img_array, verbose=0)[0]
            class_idx = np.argmax(prediction)
            confidence = float(prediction[class_idx])

            status = class_names[class_idx]
            label = f"{status} ({confidence * 100:.1f}%)"

            # Emerald green for Fresh (idx 0), Red for Rotten (idx 1)
            color = (129, 185, 16) if class_idx == 0 else (68, 68, 239)

            # Draw bounding box and label for EACH fruit
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_img, (x1, max(0, y1 - h_text - 12)), (x1 + w_text + 8, max(h_text + 12, y1)), color, -1)
            cv2.putText(annotated_img, label, (x1 + 4, max(h_text + 4, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detected_items.append({
                'status': status,
                'confidence': round(confidence * 100, 2),
                'box': [x1, y1, x2, y2]
            })

    # Fallback to full image if no specific produce crops were found
    if len(detected_items) == 0:
        resized = cv2.resize(img, (224, 224))
        img_array = img_to_array(resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)[0]
        class_idx = np.argmax(prediction)
        confidence = float(prediction[class_idx])
        status = class_names[class_idx]

        label = f"{status} ({confidence * 100:.1f}%)"
        color = (129, 185, 16) if class_idx == 0 else (68, 68, 239)

        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated_img, (15, 15), (25 + w_text, 25 + h_text + 10), color, -1)
        cv2.putText(annotated_img, label, (20, 20 + h_text + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        detected_items.append({
            'status': status,
            'confidence': round(confidence * 100, 2),
            'box': [0, 0, img_w, img_h]
        })

    # Compute overall status & summary statistics
    fresh_items = [item for item in detected_items if item['status'] == 'Fresh']
    rotten_items = [item for item in detected_items if item['status'] == 'Rotten']

    fresh_count = len(fresh_items)
    rotten_count = len(rotten_items)
    total_count = len(detected_items)

    fresh_pct = round((fresh_count / total_count) * 100, 1) if total_count > 0 else 0.0
    rotten_pct = round((rotten_count / total_count) * 100, 1) if total_count > 0 else 0.0

    if rotten_count == 0:
        overall_status = 'Fresh'
        avg_confidence = float(np.mean([item['confidence'] for item in fresh_items])) if fresh_items else 0.0
        result_summary_str = f"Fresh ({avg_confidence:.1f}%)"
    elif fresh_count == 0:
        overall_status = 'Rotten'
        avg_confidence = float(np.mean([item['confidence'] for item in rotten_items])) if rotten_items else 0.0
        result_summary_str = f"Rotten ({avg_confidence:.1f}%)"
    else:
        overall_status = 'Mixed'
        avg_confidence = float(np.mean([item['confidence'] for item in detected_items]))
        result_summary_str = f"Mixed ({fresh_pct:.0f}% Fresh, {rotten_pct:.0f}% Rotten)"

    # Encode annotated image directly into Base64 Data URL string in RAM
    _, buffer = cv2.imencode('.jpg', annotated_img)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    image_data_url = f"data:image/jpeg;base64,{base64_image}"

    # Return JSON response if requested by API call
    if wants_json:
        return jsonify({
            'success': True,
            'status': overall_status,
            'confidence': round(avg_confidence, 2),
            'prediction': result_summary_str,
            'detection_count': total_count,
            'fresh_count': fresh_count,
            'rotten_count': rotten_count,
            'fresh_pct': fresh_pct,
            'rotten_pct': rotten_pct,
            'detections': detected_items,
            'image_url': image_data_url
        })

    # Render HTML template response
    return render_template(
        'result.html',
        image_path=image_data_url,
        prediction=result_summary_str,
        overall_status=overall_status,
        confidence=round(avg_confidence, 2),
        detection_count=total_count,
        fresh_count=fresh_count,
        rotten_count=rotten_count,
        fresh_pct=fresh_pct,
        rotten_pct=rotten_pct
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
