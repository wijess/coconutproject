import os
import random
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (

    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2]
)

eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # used for val AND test arrays

dataset_dir = "dataset_clean"
classes = ["coconut", "non_coconut"]
output_dir = "stratified_kfold_split"

image_paths = []
labels = []

for label, class_name in enumerate(classes):
    class_folder = os.path.join(dataset_dir, class_name)
    for img in sorted(os.listdir(class_folder)):
        image_paths.append(os.path.join(class_folder, img))
        labels.append(label)

image_paths = np.array(image_paths)
labels = np.array(labels)

print(f"✅ Total images : {len(image_paths)}")
print(f"   Coconut      : {np.sum(labels == 0)}")
print(f"   Non-Coconut  : {np.sum(labels == 1)}")

def load_images(indices, image_paths, labels, img_size):
    imgs, lbls = [], []
    for i in indices:
        img = cv2.imread(image_paths[i])
        if img is None:
            print(f"⚠️  Skipping unreadable file: {image_paths[i]}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        imgs.append(img)
        lbls.append(labels[i])
    return np.array(imgs, dtype="float32"), np.array(lbls)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

IMG_SIZE = 224
BATCH_SIZE = 16
accuracies = []
all_true_labels = []
all_pred_labels = []
best_val_accuracy = 0.0
best_fold = -1
best_history1 = None
best_history2 = None
BEST_MODEL_PATH = "coconut_finetuned_best.keras"

for fold, (train_idx, test_idx) in enumerate(skf.split(image_paths, labels)):

    print(f"\n{'=' * 40}")
    print(f"  Fold {fold + 1} / {skf.n_splits}")
    print(f"{'=' * 40}")

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    train_idx_inner, val_idx_inner = train_test_split(
        train_idx, test_size=0.1, stratify=labels[train_idx], random_state=42
    )

    train_images, train_labels_arr = load_images(train_idx_inner, image_paths, labels, IMG_SIZE)
    val_images, val_labels_arr = load_images(val_idx_inner, image_paths, labels, IMG_SIZE)
    test_images, test_labels_arr = load_images(test_idx, image_paths, labels, IMG_SIZE)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels_arr),
        y=train_labels_arr
    )
    class_weight_dict = dict(enumerate(class_weights))

    val_images_preprocessed = preprocess_input(val_images.copy())

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    early_stop_p1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print(f"\n🚀 Phase 1: Frozen base training...")

    history1 = model.fit(
        train_datagen.flow(train_images, train_labels_arr, batch_size=BATCH_SIZE),
        epochs=25,
        validation_data=(val_images_preprocessed, val_labels_arr),
        callbacks=[early_stop_p1],
        class_weight=class_weight_dict,
        verbose=1
    )

    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    early_stop_p2 = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    print(f"\n🔧 Phase 2: Fine-tuning...")

    history2 = model.fit(
        train_datagen.flow(train_images, train_labels_arr, batch_size=BATCH_SIZE),
        epochs=30,
        validation_data=(val_images_preprocessed, val_labels_arr),
        callbacks=[early_stop_p2],
        class_weight=class_weight_dict,
        verbose=1
    )

    test_images_preprocessed = preprocess_input(test_images.copy())
    loss, accuracy = model.evaluate(test_images_preprocessed, test_labels_arr, verbose=0)
    val_loss_final, val_accuracy_final = model.evaluate(val_images_preprocessed, val_labels_arr, verbose=0)

    print(f"\n📊 Fold {fold + 1} Test Accuracy : {accuracy * 100:.2f}%  (Val Accuracy: {val_accuracy_final * 100:.2f}%)")
    accuracies.append(accuracy)

    y_pred_prob = model.predict(test_images_preprocessed)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    all_true_labels.extend(test_labels_arr)
    all_pred_labels.extend(y_pred)

    if val_accuracy_final > best_val_accuracy:
        best_val_accuracy = val_accuracy_final
        best_fold = fold + 1
        best_history1 = history1.history
        best_history2 = history2.history
        model.save(BEST_MODEL_PATH)
        print(f"   💾 New best model saved (Fold {fold + 1}, selected by val accuracy)")

print(f"\n{'=' * 40}")
print(f"  FINAL K-FOLD RESULTS")
print(f"{'=' * 40}")
print(f"🏆 Headline metric — Mean Test Accuracy : {np.mean(accuracies) * 100:.2f}%  (± {np.std(accuracies) * 100:.2f}%)")
print(f"💾 Saved deployment model               : Fold {best_fold} (selected by val accuracy: {best_val_accuracy * 100:.2f}%)")
print(f"\n📋 Overall Classification Report (All Folds Combined):")
print(classification_report(all_true_labels, all_pred_labels, target_names=classes,digits = 4))
print(f"\n📊 Overall Confusion Matrix:")
print(confusion_matrix(all_true_labels, all_pred_labels))

acc = best_history1['accuracy'] + best_history2['accuracy']
val_acc = best_history1['val_accuracy'] + best_history2['val_accuracy']
loss_ = best_history1['loss'] + best_history2['loss']
val_loss = best_history1['val_loss'] + best_history2['val_loss']
phase1_end = len(best_history1['accuracy'])

plt.figure(figsize=(11, 4))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(x=phase1_end - 0.5, color='red', linestyle='--', label='Phase 2 Transition')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Stratified K-Fold: Accuracy Curve (Best Fold {best_fold})')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss_, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(x=phase1_end - 0.5, color='red', linestyle='--', label='Phase 2 Transition')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Stratified K-Fold: Loss Curve (Best Fold {best_fold})')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('kfold_best_learning_curves.png', dpi=300)
print("\n📈 Training curves saved to 'stratified_kfold_curves.png'")
plt.show()