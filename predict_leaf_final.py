import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224

STAGE2A_CLASSES = ["healthy", "gray_leaf_blight", "leaf_rot"]
STAGE2B_CLASSES = ["negative", "positive"]  # negative=no deficiency, positive=deficiency present

# ---- Load models ----
stage1_model = load_model("coconut_finetuned_best.keras")
stage2a_model = load_model("stage2a_stratified_kfold_best.keras")   # disease-type (best: K-Fold)
stage2b_model = load_model("stage2b_stratified_kfold_best.keras")   # nutrient deficiency (best: K-Fold)


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img.astype("float32"), axis=0)
    return preprocess_input(img_array.copy())


def predict_pipeline(image_path):
    img_preprocessed = preprocess_image(image_path)

    print("=" * 55)
    print(f"Analyzing: {image_path}")
    print("=" * 55)

    #Coconut identification
    # coconut=0, non_coconut=1
    stage1_prob = stage1_model.predict(img_preprocessed, verbose=0)[0][0]
    is_coconut = stage1_prob < 0.5

    if not is_coconut:
        print(f"\nStage 1: REJECTED - Not a coconut leaf "
              f"({stage1_prob * 100:.2f}% non-coconut confidence)")
        return

    coconut_confidence = (1 - stage1_prob) * 100
    print(f"\nStage 1: Coconut leaf confirmed ({coconut_confidence:.2f}% confidence)")

    #2a: Disease-type classification (3-class) ----
    stage2a_probs = stage2a_model.predict(img_preprocessed, verbose=0)[0]
    predicted_idx = np.argmax(stage2a_probs)
    predicted_disease = STAGE2A_CLASSES[predicted_idx]
    disease_confidence = stage2a_probs[predicted_idx] * 100

    print(f"\nStage 2a - Disease Type: {predicted_disease}  ({disease_confidence:.2f}%)")
    print("  Full breakdown:")
    for i, cls in enumerate(STAGE2A_CLASSES):
        print(f"    {cls:20s}: {stage2a_probs[i] * 100:.2f}%")

    # Stage 2b: Nutrient Deficiency (binary) ----
    stage2b_prob = stage2b_model.predict(img_preprocessed, verbose=0)[0][0]
    if stage2b_prob > 0.5:
        deficiency_label = "nutrient deficiency detected"
        deficiency_confidence = stage2b_prob * 100
    else:
        deficiency_label = "nutrient deficiency not detected"
        deficiency_confidence = (1 - stage2b_prob) * 100

    print(f"\nStage 2b - Nutrient Deficiency: {deficiency_label}  "
          f"({deficiency_confidence:.2f}%)")


    print("\n" + "=" * 55)
    print("FINAL RESULT")
    print("=" * 55)
    print(f"Leaf Type       : Coconut ({coconut_confidence:.2f}%)")
    print(f"Disease         : {predicted_disease} ({disease_confidence:.2f}%)")
    print(f"Nutrient Status : Deficiency {deficiency_label} ({deficiency_confidence:.2f}%)")
    print("=" * 55)


if __name__ == "__main__":
    test_image_path = "diseased leaf.jpg"   # <-- change to your leaf image's filename/path
    predict_pipeline(test_image_path)