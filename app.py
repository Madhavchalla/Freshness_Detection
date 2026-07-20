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

class_names = ['Fresh', 'Rotten']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        if request.wants_json or request.headers.get('Accept') == 'application/json':
            return jsonify({'error': 'No image file found'}), 400
        return 'No image file found'

    file = request.files['image']
    if file.filename == '':
        if request.wants_json or request.headers.get('Accept') == 'application/json':
            return jsonify({'error': 'No file selected'}), 400
        return 'No file selected'

    # Read image directly from RAM buffer into OpenCV (NO DISK FILE SAVED!)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        if request.wants_json or request.headers.get('Accept') == 'application/json':
            return jsonify({'error': 'Invalid image file'}), 400
        return 'Invalid image file'

    # Run YOLO detection directly on OpenCV image array in RAM
    results = yolo_model(img)
    detections = results.xyxy[0]  # Tensor of detections

    crop_found = False
    
    # Choose highest confidence detection if available
    if len(detections) > 0:
        x1, y1, x2, y2, conf, cls = detections[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Crop localized fruit in RAM
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            resized = cv2.resize(crop, (224, 224))
            img_array = img_to_array(resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            crop_found = True
            
    # Fallback to full image if no crop was detected
    if not crop_found:
        resized = cv2.resize(img, (224, 224))
        img_array = img_to_array(resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    # Predict class indices
    prediction = model.predict(img_array)[0]
    class_idx = np.argmax(prediction)
    confidence = float(prediction[class_idx])

    # Display prediction string
    result = f"{class_names[class_idx]} ({confidence * 100:.2f}%)"

    # Draw annotations on in-memory image copy
    annotated_img = img.copy()
    color = (129, 185, 16) if class_idx == 0 else (68, 68, 239)
    label = f"{class_names[class_idx]} ({confidence * 100:.1f}%)"

    if crop_found:
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated_img, (x1, y1 - h_text - 15), (x1 + w_text + 10, y1), color, -1)
        cv2.putText(annotated_img, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated_img, (15, 15), (25 + w_text, 25 + h_text + 10), color, -1)
        cv2.putText(annotated_img, label, (20, 20 + h_text + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Encode annotated OpenCV image directly into a Base64 Data URL string in RAM
    _, buffer = cv2.imencode('.jpg', annotated_img)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    image_data_url = f"data:image/jpeg;base64,{base64_image}"

    # Return JSON response if requested by API call (Vercel Frontend)
    if request.wants_json or request.headers.get('Accept') == 'application/json' or request.is_json:
        return jsonify({
            'success': True,
            'status': class_names[class_idx],
            'confidence': round(confidence * 100, 2),
            'prediction': result,
            'image_url': image_data_url
        })

    # Standard Flask template rendering with Base64 data URL
    return render_template('result.html', image_path=image_data_url, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
