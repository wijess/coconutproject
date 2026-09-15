import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMAGE_PATH = "test_images/BudRot397.jpg"

MODEL_PATH = "coconut_finetuned_best.keras"
IMG_SIZE = 224
CLASSES = ["coconut", "non_coconut"]           # label 0 = coconut, 1 = non_coconut
THRESHOLD = 0.5

print(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)


def load_and_preprocess(img_path):

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def predict_image(img_path):
    img_array = load_and_preprocess(img_path)
    prob = float(model.predict(img_array, verbose=0)[0][0])

    class_idx = 1 if prob >= THRESHOLD else 0
    confidence = prob if class_idx == 1 else 1 - prob
    label = CLASSES[class_idx]

    return label, confidence, prob

if __name__ == "__main__":
    label, confidence, raw_prob = predict_image(IMAGE_PATH)

    print(f"\nImage       : {IMAGE_PATH}")
    print(f"Prediction  : {label}")
    print(f"Confidence  : {confidence * 100:.2f}%")
    print(f"Raw sigmoid : {raw_prob:.4f}")