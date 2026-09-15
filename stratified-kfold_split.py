import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

dataset_dir = "dataset_clean"
classes = ["coconut", "non_coconut"]

image_paths = []
labels = []

for label, class_name in enumerate(classes):
    class_folder = os.path.join(dataset_dir, class_name)

    for img in os.listdir(class_folder):
        image_paths.append(os.path.join(class_folder, img))
        labels.append(label)

image_paths = np.array(image_paths)
labels = np.array(labels)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Stratified K-Fold Split Ready")
