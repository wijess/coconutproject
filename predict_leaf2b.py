import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224

# ---- Load models ----
stage1_model = load_model("coconut_finetuned_best.keras")          # Stage 1

# Pick ONE of these for Stage 2b (or run both to compare, see below)
stage2b_random_model = load_model("stage2b_random_split_model.keras")
stage2b_kfold_model = load_model("stage2b_stratified_kfold_best.keras")  # once k-fold finishes


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img.astype("float32"), axis=0)
    return preprocess_input(img_array.copy())


def predict_stage2b(model, img_preprocessed):
    """Returns (label, confidence_percent) for the nutrient-deficiency model."""
    prob = model.predict(img_preprocessed, verbose=0)[0][0]
    # label order in training: ["negative", "positive"] -> 0=negative, 1=positive
    if prob > 0.5:
        return "Nutrient Deficiency: PRESENT", prob * 100
    else:
        return "Nutrient Deficiency: ABSENT", (1 - prob) * 100


def predict_pipeline(image_path):
    img_preprocessed = preprocess_image(image_path)

    # Coconut identification ----
    stage1_prob = stage1_model.predict(img_preprocessed, verbose=0)[0][0]
    is_coconut = stage1_prob < 0.5  # adjust based on your label encoding (0=coconut)

    if not is_coconut:
        print("Stage 1 Result: REJECTED - Not a coconut leaf")
        print(f"  Confidence: {(1 - stage1_prob) * 100:.2f}%")
        return

    print(f"Stage 1 Result: Coconut leaf confirmed ({stage1_prob*100:.2f}% coconut confidence)\n")

    #2b: Nutrient Deficiency (Random Split model) ----
    label_rs, conf_rs = predict_stage2b(stage2b_random_model, img_preprocessed)
    print(f"[Random Split Model]      {label_rs}  ({conf_rs:.2f}%)")

    #2b: Nutrient Deficiency (Stratified K-Fold model) ----
    try:
        label_kf, conf_kf = predict_stage2b(stage2b_kfold_model, img_preprocessed)
        print(f"[Stratified K-Fold Model] {label_kf}  ({conf_kf:.2f}%)")
    except Exception as e:
        print(f"[Stratified K-Fold Model] Not available yet ({e})")

    print("\nNOTE: Disease-type (gray_leaf_blight / leaf_rot / healthy) "
          "classification is not yet available -- Stage 2a model pending.")


# ---- Example usage ----
if __name__ == "__main__":
    test_image_path = "diseased leaf.jpg"   # <-- change to your test image path
    predict_pipeline(test_image_path)
