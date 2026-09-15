import os
import random
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

# Seeds (Reproducibility)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

input_dir = "dataset_clean"
output_dir = "kfold_split"
classes = ["coconut", "non_coconut"]
k = 5

image_paths, labels = [], []
for label, c in enumerate(classes):
    class_folder = os.path.join(input_dir, c)
    for img in sorted(os.listdir(class_folder)):
        image_paths.append((c, img))
        labels.append(label)

image_paths = np.array(image_paths, dtype=object)
labels = np.array(labels)

skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create Directory Structure
for fold, (train_idx, test_idx) in enumerate(skf.split(image_paths, labels), start=1):
    for split in ["train", "test"]:
        for c in classes: os.makedirs(os.path.join(output_dir, f"fold{fold}", split, c), exist_ok=True)
    for i in train_idx:
        c, img = image_paths[i]
        shutil.copy(os.path.join(input_dir, c, img), os.path.join(output_dir, f"fold{fold}", "train", c, img))
    for i in test_idx:
        c, img = image_paths[i]
        shutil.copy(os.path.join(input_dir, c, img), os.path.join(output_dir, f"fold{fold}", "test", c, img))

IMG_SIZE, BATCH_SIZE = (224, 224), 16
accuracies = []
best_accuracy = 0.0
best_fold = -1

# Arrays to accumulate all predictions and ground truths across folds
all_true_labels = []
all_pred_labels = []

# To store the training history of the best performing fold
best_history1 = None
best_history2 = None

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30, zoom_range=0.3, horizontal_flip=True, vertical_flip=True,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, brightness_range=[0.8, 1.2]
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Fold Training Loop
for fold in range(1, k + 1):
    print(f"\n--- Fold {fold} ---")
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    train_generator = train_datagen.flow_from_directory(os.path.join(output_dir, f"fold{fold}/train"),
                                                        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                        class_mode="binary", shuffle=True, seed=42)
    test_generator = test_datagen.flow_from_directory(os.path.join(output_dir, f"fold{fold}/test"),
                                                      target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary",
                                                      shuffle=False)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes),
                                         y=train_generator.classes)
    class_weight_dict = dict(enumerate(class_weights))

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model, GlobalAveragePooling2D(), BatchNormalization(),
        Dense(256, activation='relu'), Dropout(0.5),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Phase 1
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    early_stop_p1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history1 = model.fit(train_generator, epochs=25, validation_data=test_generator, callbacks=[early_stop_p1],
                         class_weight=class_weight_dict, verbose=1)

    # Phase 2
    base_model.trainable = True
    for layer in base_model.layers[:-50]: layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    early_stop_p2 = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    train_generator.reset()
    history2 = model.fit(train_generator, epochs=30, validation_data=test_generator, callbacks=[early_stop_p2],
                         class_weight=class_weight_dict, verbose=1)

    loss, accuracy = model.evaluate(test_generator, verbose=0)
    accuracies.append(accuracy)
    print(f"Fold {fold} Accuracy : {accuracy * 100:.2f}%")

    # Generate predictions for the current fold
    test_generator.reset()
    y_true_fold = test_generator.classes
    y_pred_prob_fold = model.predict(test_generator)
    y_pred_fold = (y_pred_prob_fold > 0.5).astype("int32").flatten()

    # Store predictions and truths for global metrics calculation later
    all_true_labels.extend(y_true_fold)
    all_pred_labels.extend(y_pred_fold)

    # Correct Logic: Save the overall best model across folds & capture its history
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_fold = fold
        best_history1 = history1.history
        best_history2 = history2.history
        model.save("best_directory_kfold.keras")
        print(f"   💾 New best model saved from Fold {fold}")

# ─────────────────────────────────────────────
# 6. FINAL INTER-FOLD RESULTS (EVALUATION METRIC)
# ─────────────────────────────────────────────
print(f"\n{'=' * 40}")
print(f"  FINAL DIRECTORY K-FOLD RESULTS")
print(f"{'=' * 40}")
print(f"🏆 Best Fold           : Fold {best_fold} (Accuracy: {best_accuracy * 100:.2f}%)")
print(f"🏆 Mean Accuracy       : {np.mean(accuracies) * 100:.2f}%")
print(f"📉 Std Deviation       : {np.std(accuracies) * 100:.2f}%")

# Generate Global Metrics across all 5 Folds
cm = confusion_matrix(all_true_labels, all_pred_labels)
print("\n📊 Overall Confusion Matrix (All Folds Combined):\n", cm)

precision = precision_score(all_true_labels, all_pred_labels)
recall = recall_score(all_true_labels, all_pred_labels)
f1 = f1_score(all_true_labels, all_pred_labels)

print(f"\nOverall Precision : {precision:.4f}")
print(f"Overall Recall    : {recall:.4f}")
print(f"Overall F1 Score  : {f1:.4f}")
print("\n📋 Overall Classification Report:\n",
      classification_report(all_true_labels, all_pred_labels, target_names=classes))

# ─────────────────────────────────────────────
# 7. PLOT TRAINING CURVES FOR THE BEST FOLD
# ─────────────────────────────────────────────
acc = best_history1['accuracy'] + best_history2['accuracy']
val_acc = best_history1['val_accuracy'] + best_history2['val_accuracy']
loss_ = best_history1['loss'] + best_history2['loss']
val_loss = best_history1['val_loss'] + best_history2['val_loss']
phase1_end = len(best_history1['accuracy'])

plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy', color='blue')
plt.plot(val_acc, label='Validation Accuracy', color='orange')
plt.axvline(x=phase1_end - 1, color='red', linestyle='--', label='Phase 2 Transition')
plt.title(f'Directory K-Fold: Accuracy Curve (Best Fold {best_fold})')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(loss_, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='orange')
plt.axvline(x=phase1_end - 1, color='red', linestyle='--', label='Phase 2 Transition')
plt.title(f'Directory K-Fold: Loss Curve (Best Fold {best_fold})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('directory_kfold_learning_curves.png', dpi=300)
print("\n[INFO] Best fold learning curves saved as 'directory_kfold_learning_curves.png'")
plt.show()