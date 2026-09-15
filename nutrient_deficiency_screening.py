import cv2
import numpy as np
import os
import csv
import shutil

SOURCE_DIR = "dataset_clean_disease"
NUTRIENT_DEFICIENCY_FOLDER = os.path.join(SOURCE_DIR, "nutrient_deficiency")
NEGATIVE_CANDIDATE_CLASSES = ["gray_leaf_blight", "leaf_rot", "healthy"]

STAGE2B_OUTPUT_DIR = "dataset_stage2b"
POSITIVE_OUT = os.path.join(STAGE2B_OUTPUT_DIR, "positive")   # nutrient deficiency present
NEGATIVE_OUT = os.path.join(STAGE2B_OUTPUT_DIR, "negative")   # nutrient deficiency absent

CSV_LOG_PATH = "nutrient_deficiency_screening.csv"

# Yellow/chlorosis HSV range -- tune if your dataset's lighting/camera differs
LOWER_YELLOW = np.array([15, 40, 40])
UPPER_YELLOW = np.array([35, 255, 255])

# Threshold strategy: mean - 1*std of the confirmed nutrient_deficiency class.
# This is a conservative cutoff -- images with a yellow-ratio at or above
# this value are flagged as "possible co-occurring nutrient deficiency".
STD_MULTIPLIER = 1.0


def calculate_yellowing_ratio(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    yellow_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return yellow_ratio


def main():
    os.makedirs(POSITIVE_OUT, exist_ok=True)
    os.makedirs(NEGATIVE_OUT, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Build baseline distribution from the CONFIRMED
    # nutrient_deficiency class
    # -------------------------------------------------------------
    print("Step 1: Calculating baseline yellow-ratio distribution "
          "from nutrient_deficiency class...")

    nd_ratios = []
    nd_images = sorted(os.listdir(NUTRIENT_DEFICIENCY_FOLDER))
    for img_name in nd_images:
        ratio = calculate_yellowing_ratio(os.path.join(NUTRIENT_DEFICIENCY_FOLDER, img_name))
        if ratio is not None:
            nd_ratios.append(ratio)

    nd_mean = np.mean(nd_ratios)
    nd_std = np.std(nd_ratios)
    threshold = nd_mean - (STD_MULTIPLIER * nd_std)

    print(f"  Nutrient Deficiency class: n={len(nd_ratios)}, "
          f"mean={nd_mean:.4f}, std={nd_std:.4f}")
    print(f"  Flagging threshold        : {threshold:.4f}\n")

    # -------------------------------------------------------------
    # Step 2: Screen the negative-candidate classes
    # -------------------------------------------------------------
    print("Step 2: Screening disease-type classes for possible "
          "co-occurring nutrient deficiency...\n")

    log_rows = []
    class_summary = {}

    for c in NEGATIVE_CANDIDATE_CLASSES:
        folder = os.path.join(SOURCE_DIR, c)
        img_names = sorted(os.listdir(folder))

        flagged_count = 0
        passed_count = 0

        for img_name in img_names:
            img_path = os.path.join(folder, img_name)
            ratio = calculate_yellowing_ratio(img_path)
            if ratio is None:
                continue

            is_flagged = ratio >= threshold
            log_rows.append([c, img_name, f"{ratio:.4f}", is_flagged])

            if is_flagged:
                flagged_count += 1
                # Excluded -- NOT copied to the Stage 2b negative set
            else:
                passed_count += 1
                try:
                    shutil.copy(img_path, os.path.join(NEGATIVE_OUT, f"{c}_{img_name}"))
                except (PermissionError, OSError):
                    pass  # counted as passed/screened, but copy retried in a rerun if needed

        total = flagged_count + passed_count
        pct_flagged = (flagged_count / total * 100) if total else 0
        class_summary[c] = (total, flagged_count, passed_count, pct_flagged)

        print(f"  {c:20s}: total={total:5d}  flagged={flagged_count:5d} "
              f"({pct_flagged:5.1f}%)  passed={passed_count:5d}")

    # -------------------------------------------------------------
    # Step 3: Copy positive class (nutrient_deficiency) as-is
    # -------------------------------------------------------------
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
        print(f"  WARNING: {len(copy_failures)} files failed to copy "
              f"(locked/permission issues). See copy_failures.log")
        with open("copy_failures.log", "w") as f:
            for name, err in copy_failures:
                f.write(f"{name}: {err}\n")

    # -------------------------------------------------------------
    # Step 4: Save CSV log
    # -------------------------------------------------------------
    with open(CSV_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "image", "yellow_ratio", "flagged_as_contaminated"])
        writer.writerows(log_rows)

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------
    total_negative_final = sum(v[2] for v in class_summary.values())
    total_positive_final = len(nd_images)

    print(f"\n{'='*50}")
    print("SCREENING COMPLETE")
    print(f"{'='*50}")
    print(f"Positive class (nutrient_deficiency): {total_positive_final} images  -> {POSITIVE_OUT}")
    print(f"Negative class (screened, contamination-free): {total_negative_final} images  -> {NEGATIVE_OUT}")
    print(f"Full per-image log saved to: {CSV_LOG_PATH}")
    print(f"\nClass imbalance (Stage 2b): "
          f"{total_positive_final} positive vs {total_negative_final} negative "
          f"(ratio {total_positive_final/total_negative_final:.2f}x)")
    print("\nNext: use dataset_stage2b/positive and dataset_stage2b/negative "
          "as the two classes for the Stage 2b binary training script.")


if __name__ == "__main__":
    main()
