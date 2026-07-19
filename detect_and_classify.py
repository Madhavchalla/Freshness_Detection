# import torch
# import cv2
# import os
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import numpy as np
#
# # Load YOLOv5 model
# model_yolo = torch.hub.load('yolov5', 'yolov5s', source='local')
# # Load your classifier model
# model_cls = load_model('healthy_vs_rotten.h5')
# class_names = ['Fresh', 'Rotten']
#
# def classify_fruits(image_path):
#     results = model_yolo(image_path)
#     labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
#
#     img = cv2.imread(image_path)
#     h, w, _ = img.shape
#     annotated_img = img.copy()
#
#     for i in range(len(labels)):
#         x1, y1, x2, y2, conf = coords[i]
#         x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
#
#         crop = img[y1:y2, x1:x2]
#         if crop.size == 0:
#             continue
#
#         crop_resized = cv2.resize(crop, (224, 224))
#         crop_array = img_to_array(crop_resized) / 255.0
#         crop_array = np.expand_dims(crop_array, axis=0)
#
#         pred = model_cls.predict(crop_array)[0]
#         cls_index = np.argmax(pred)
#         label = f"{class_names[cls_index]} ({pred[cls_index]*100:.1f}%)"
#
#         # Draw results
#         cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0,255,0), 2)
#         cv2.putText(annotated_img, label, (x1, y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#
#     # Save result
#     output_path = 'static/result.jpg'
#     os.makedirs('static', exist_ok=True)
#     cv2.imwrite(output_path, annotated_img)
#     return output_path

import torch
from pathlib import Path
from ultralytics import YOLO
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load YOLOv5 model from local path
yolo_model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')  # Or your trained .pt model

# Load your Keras classification model
classifier = load_model('healthy_vs_rotten.h5')

# Class names your Keras model was trained on (update accordingly)
class_names = ['fresh', 'rotten']

# Function to classify a single fruit image
def classify_crop(crop_img):
    resized = cv2.resize(crop_img, (224, 224))  # match model input size
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 224, 224, 3))
    prediction = classifier.predict(reshaped)[0]
    return class_names[np.argmax(prediction)]

# Input image
image_path = 'test_images/sample.jpg'  # Change to your image path
img = cv2.imread(image_path)

# Perform detection
results = yolo_model(image_path)

for result in results.xyxy[0]:  # xyxy[0] gives detections
    x1, y1, x2, y2, conf, cls = result
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    fruit_crop = img[y1:y2, x1:x2]

    label = classify_crop(fruit_crop)

    # Draw bounding box and label
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 0, 0), 2)

# Show output image
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

