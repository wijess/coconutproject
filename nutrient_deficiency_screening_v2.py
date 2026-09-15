import cv2
import numpy as np
import os
import csv
import shutil

# CONFIG
SOURCE_DIR = "dataset_clean_disease"
NUTRIENT_DEFICIENCY_FOLDER = os.path.join(SOURCE_DIR, "nutrient_deficiency")
NEGATIVE_CANDIDATE_CLASSES = ["gray_leaf_blight", "leaf_rot", "healthy"]

STAGE2B_OUTPUT_DIR = "dataset_stage2b"
POSITIVE_OUT = os.path.join(STAGE2B_OUTPUT_DIR, "positive")
NEGATIVE_OUT = os.path.join(STAGE2B_OUTPUT_DIR, "negative")

CSV_LOG_PATH = "nutrient_deficiency_screening_v2.csv"

# --- HSV ranges for the three symptom cues ---
# Yellow
LOWER_YELLOW = np.array([15, 40, 40])
UPPER_YELLOW = np.array([35, 255, 255])

# Brown/necrosis (marginal browning, tip dieback -- potassium deficiency)
LOWER_BROWN = np.array([5, 50, 20])
UPPER_BROWN = np.array([20, 255, 150])

# Dark/black spotting (black-seed discoloration)
LOWER_DARK = np.array([0, 0, 0])
UPPER_DARK = np.array([180, 255, 50])

STD_MULTIPLIER = 1.0


def calculate_cue_ratios(image_path):
    """Returns (yellow_ratio, brown_ratio, dark_ratio) for one image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_px = hsv.shape[0] * hsv.shape[1]

    yellow_mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    brown_mask = cv2.inRange(hsv, LOWER_BROWN, UPPER_BROWN)
    dark_mask = cv2.inRange(hsv, LOWER_DARK, UPPER_DARK)

    yellow_ratio = np.sum(yellow_mask > 0) / total_px
    brown_ratio = np.sum(brown_mask > 0) / total_px
    dark_ratio = np.sum(dark_mask > 0) / total_px

    return yellow_ratio, brown_ratio, dark_ratio


def main():
    os.makedirs(POSITIVE_OUT, exist_ok=True)
    os.makedirs(NEGATIVE_OUT, exist_ok=True)


    # Step 1: Build baseline distributions (per cue) from the
    # CONFIRMED nutrient_deficiency class
    print("Step 1: Calculating baseline cue distributions from "
          "nutrient_deficiency class...")

    nd_images = sorted(os.listdir(NUTRIENT_DEFICIENCY_FOLDER))
    yellow_vals, brown_vals, dark_vals = [], [], []

    for img_name in nd_images:
        result = calculate_cue_ratios(os.path.join(NUTRIENT_DEFICIENCY_FOLDER, img_name))
        if result is None:
            continue
        y, b, d = result
        yellow_vals.append(y)
        brown_vals.append(b)
        dark_vals.append(d)

    yellow_thresh = max(0.02, np.mean(yellow_vals) - STD_MULTIPLIER * np.std(yellow_vals))
    brown_thresh = max(0.02, np.mean(brown_vals) - STD_MULTIPLIER * np.std(brown_vals))
    dark_thresh = max(0.02, np.mean(dark_vals) - STD_MULTIPLIER * np.std(dark_vals))

    print(f"  Yellow: mean={np.mean(yellow_vals):.4f} std={np.std(yellow_vals):.4f} "
          f"threshold={yellow_thresh:.4f}")
    print(f"  Brown : mean={np.mean(brown_vals):.4f} std={np.std(brown_vals):.4f} "
          f"threshold={brown_thresh:.4f}")
    print(f"  Dark  : mean={np.mean(dark_vals):.4f} std={np.std(dark_vals):.4f} "
          f"threshold={dark_thresh:.4f}\n")

    # Step 2: Screen the negative-candidate classes
    # An image is flagged if it exceeds ANY of the three thresholds
    # (i.e. shows a strong signal on at least one symptom cue)
    print("Step 2: Screening disease-type classes (multi-cue)...\n")

    log_rows = []
    class_summary = {}

    for c in NEGATIVE_CANDIDATE_CLASSES:
        folder = os.path.join(SOURCE_DIR, c)
        img_names = sorted(os.listdir(folder))

        flagged_count = 0
        passed_count = 0

        for img_name in img_names:
            img_path = os.path.join(folder, img_name)
            result = calculate_cue_ratios(img_path)
            if result is None:
                continue
            y, b, d = result

            flagged_yellow = y >= yellow_thresh
            flagged_brown = b >= brown_thresh
            flagged_dark = d >= dark_thresh
            is_flagged = flagged_yellow or flagged_brown or flagged_dark

            log_rows.append([c, img_name, f"{y:.4f}", f"{b:.4f}", f"{d:.4f}",
                              flagged_yellow, flagged_brown, flagged_dark, is_flagged])

            if is_flagged:
                flagged_count += 1
            else:
                passed_count += 1
                try:
                    shutil.copy(img_path, os.path.join(NEGATIVE_OUT, f"{c}_{img_name}"))
                except (PermissionError, OSError):
                    pass

        total = flagged_count + passed_count
        pct_flagged = (flagged_count / total * 100) if total else 0
        class_summary[c] = (total, flagged_count, passed_count, pct_flagged)

        print(f"  {c:20s}: total={total:5d}  flagged={flagged_count:5d} "
              f"({pct_flagged:5.1f}%)  passed={passed_count:5d}")


    # Step 3: Copy positive class as-is
    print("\nStep 3: Copying confirmed nutrient_deficiency images "
          "as the positive class...")

    copy_failures = []
    for img_name in nd_images:
        src = os.path.join(NUTRIENT_DEFICIENCY_FOLDER, img_name)
        dst = os.path.join(POSITIVE_OUT, img_name)
        try:
            shutil.copy(src, dst)
        except (PermissionError, OSError) as e:
            copy_failures.append((img_name, str(e)))

    if copy_failures:
        print(f"  WARNING: {len(copy_failures)} files failed to copy. "
              f"See copy_failures.log")
        with open("copy_failures.log", "w") as f:
            for name, err in copy_failures:
                f.write(f"{name}: {err}\n")

    # Step 4: Save CSV log
    with open(CSV_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "image", "yellow_ratio", "brown_ratio", "dark_ratio",
                          "flagged_yellow", "flagged_brown", "flagged_dark", "flagged_overall"])
        writer.writerows(log_rows)

#sumary
    total_negative_final = sum(v[2] for v in class_summary.values())
    total_positive_final = len(nd_images)

    print(f"\n{'='*50}")
    print("MULTI-CUE SCREENING COMPLETE")
    print(f"{'='*50}")
    print(f"Positive class (nutrient_deficiency): {total_positive_final} images  -> {POSITIVE_OUT}")
    print(f"Negative class (screened, contamination-free): {total_negative_final} images  -> {NEGATIVE_OUT}")
    print(f"Full per-image log saved to: {CSV_LOG_PATH}")
    print(f"\nClass imbalance (Stage 2b): "
          f"{total_positive_final} positive vs {total_negative_final} negative "
          f"(ratio {total_positive_final/total_negative_final:.2f}x)" if total_negative_final > 0
          else f"\nWARNING: negative class is empty (0 images) -- check thresholds/logic before proceeding.")
    print("\nNOTE: Boron-deficiency symptoms (structural malformation) are not "
          "captured by any of the three color cues and remain an undetected "
          "risk -- disclose this as a limitation.")


if __name__ == "__main__":
    main()