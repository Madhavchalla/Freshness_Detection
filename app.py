import base64
import numpy as np
import cv2
import torch
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load models at startup
print("Loading classification model (healthy_vs_rotten.h5)...")
model = load_model("healthy_vs_rotten.h5")

print("Loading YOLOv5 detection model...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

# Configure multi-object detection sensitivity
yolo_model.conf = 0.15  # Detection confidence threshold
yolo_model.iou = 0.45   # IoU threshold for overlapping items

class_names = ['Fresh', 'Rotten']

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

    # Run YOLO detection directly on OpenCV image array in RAM
    results = yolo_model(img)
    detections = results.xyxy[0]  # Tensor of ALL detected bounding boxes

    annotated_img = img.copy()
    detected_items = []
    
    img_h, img_w, _ = img.shape
    total_img_area = img_h * img_w
    
    # Process ALL detected fruits in the image
    if len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            box_area = (x2 - x1) * (y2 - y1)
            
            # Ignore full-frame outer background boxes (covering >85% of total image)
            if box_area > (total_img_area * 0.85):
                continue
                
            # Crop localized fruit in RAM
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

    # Fallback to full image if no specific YOLO fruit crops were found
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

    # Compute overall status & summary
    overall_rotten = any(item['status'] == 'Rotten' for item in detected_items)
    overall_status = 'Rotten' if overall_rotten else 'Fresh'
    avg_confidence = np.mean([item['confidence'] for item in detected_items])
    detection_count = len(detected_items)

    result_summary_str = f"{overall_status} ({avg_confidence:.2f}%)"

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
            'detection_count': detection_count,
            'detections': detected_items,
            'image_url': image_data_url
        })

    # Render HTML template response
    return render_template(
        'result.html',
        image_path=image_data_url,
        prediction=result_summary_str,
        detection_count=detection_count
    )

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
