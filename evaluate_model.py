import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

def evaluate():
    model_path = 'healthy_vs_rotten.h5'
    test_dir = 'final_dataset/test'

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}")
        return

    if not os.path.exists(test_dir):
        print(f"[ERROR] Test dataset directory not found at {test_dir}")
        return

    print("[INFO] Loading model...")
    model = load_model(model_path)

    print("[INFO] Preparing test dataset generator...")
    datagen = ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    class_labels = list(test_generator.class_indices.keys())
    print(f"[INFO] Class mapping: {test_generator.class_indices}")

    print("[INFO] Running predictions on test dataset...")
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    print("\n" + "="*55)
    print(" EVALUATION METRICS REPORT")
    print("="*55)

    # Calculate overall accuracy & loss
    loss, accuracy = model.evaluate(test_generator, verbose=0)
    print(f"\nOverall Test Accuracy : {accuracy * 100:.2f}%")
    print(f"Test Loss             : {loss:.4f}\n")

    # Detailed Classification Report (Precision, Recall, F1-Score)
    print("Classification Report:")
    print("-" * 55)
    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    print(report)

    # Confusion Matrix
    print("Confusion Matrix:")
    print("-" * 55)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Classes: {class_labels}")
    print(cm)
    
    if len(class_labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        print(f"\n- True Negatives ({class_labels[0]})  : {tn}")
        print(f"- False Positives (Misclassified as {class_labels[1]}) : {fp}")
        print(f"- False Negatives (Misclassified as {class_labels[0]}) : {fn}")
        print(f"- True Positives ({class_labels[1]})  : {tp}")

    print("\n" + "="*55)

if __name__ == '__main__':
    evaluate()
