import os
import cv2
import shutil
import hashlib
import numpy as np

input_dir = "dataset_raw_disease"

clean_dir = "dataset_clean_disease"
reject_dir = "rejected_disease"
review_dir = "candidates_for_review_disease"

classes = ["gray_leaf_blight", "healthy", "leaf_rot", "nutrient_deficiency"]

BLUR_THRESHOLD = 100
GREEN_RATIO_THRESHOLD = 0.05  # lowered from 0.15 — diseased/yellowed/dried leaves
                               # can have low green ratio and shouldn't be wrongly rejected

HAMMING_THRESHOLDS = {
    "gray_leaf_blight": 5,
    "healthy": 3,
    "leaf_rot": 5,
    "nutrient_deficiency": 5
}

for c in classes:
    for folder in [clean_dir, reject_dir, review_dir]:
        os.makedirs(os.path.join(folder, c), exist_ok=True)

def get_image_hash(image_path):
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_perceptual_hash(image, hash_size=8):
    resized = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    return diff.flatten()


def hamming_distance(hash1, hash2):
    return np.count_nonzero(hash1 != hash2)

def has_valid_leaf_content(image, green_ratio_threshold=GREEN_RATIO_THRESHOLD):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 30, 20])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    leaf_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return leaf_ratio >= green_ratio_threshold

def is_blurry(image, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

summary = {}

for c in classes:

    class_path = os.path.join(input_dir, c)
    threshold = HAMMING_THRESHOLDS[c]

    md5_hashes = set()
    phashes = []

    counts = {"clean": 0, "exact_dup": 0, "near_dup_flagged": 0, "background_reject": 0, "blur": 0}

    for img_name in sorted(os.listdir(class_path)):

        img_path = os.path.join(class_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        img_hash = get_image_hash(img_path)
        if img_hash in md5_hashes:
            shutil.copy(img_path, os.path.join(reject_dir, c, img_name))
            counts["exact_dup"] += 1
            continue
        md5_hashes.add(img_hash)

        phash = get_perceptual_hash(image)
        is_near_dup = any(hamming_distance(phash, h) <= threshold for h in phashes)
        if is_near_dup:
            shutil.copy(img_path, os.path.join(review_dir, c, img_name))
            counts["near_dup_flagged"] += 1
            phashes.append(phash)
            continue
        phashes.append(phash)

        if not has_valid_leaf_content(image):
            shutil.copy(img_path, os.path.join(reject_dir, c, img_name))
            counts["background_reject"] += 1
            continue

        if is_blurry(image):
            shutil.copy(img_path, os.path.join(reject_dir, c, img_name))
            counts["blur"] += 1
            continue

        shutil.copy(img_path, os.path.join(clean_dir, c, img_name))
        counts["clean"] += 1

    summary[c] = counts

print("Dataset cleaning finished!\n")

for c in classes:
    print("--------------------------------")
    print(c)
    print("Clean images          :", summary[c]["clean"])
    print("Exact duplicates       :", summary[c]["exact_dup"])
    print("Near-duplicates flagged:", summary[c]["near_dup_flagged"], "(review manually in candidates_for_review_disease/)")
    print("Background rejects     :", summary[c]["background_reject"])
    print("Blur/outliers          :", summary[c]["blur"])

print("\n[INFO] Review 'candidates_for_review_disease/' manually (Explorer thumbnail view).")
print("[INFO] Confirmed duplicates: delete from dataset_clean_disease/ if a copy exists there,")
print("[INFO] or leave in candidates_for_review_disease/ if it was never added to dataset_clean_disease/.")