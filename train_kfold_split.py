import os
import random
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split   # CHANGED: KFold, not StratifiedKFold (see note below)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

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

kf = KFold(n_splits=k, shuffle=True, random_state=42)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

for fold, (train_idx, test_idx) in enumerate(kf.split(image_paths), start=1):   # CHANGED: kf.split(image_paths) — no labels arg
    train_idx_inner, val_idx_inner = train_test_split(
        train_idx, test_size=0.1, random_state=42
    )

    for split in ["train", "val", "test"]:
        for c in classes:
            os.makedirs(os.path.join(output_dir, f"fold{fold}", split, c), exist_ok=True)

    for i in train_idx_inner:
        c, img = image_paths[i]
        shutil.copy(os.path.join(input_dir, c, img), os.path.join(output_dir, f"fold{fold}", "train", c, img))
    for i in val_idx_inner:
        c, img = image_paths[i]
        shutil.copy(os.path.join(input_dir, c, img), os.path.join(output_dir, f"fold{fold}", "val", c, img))
    for i in test_idx:
        c, img = image_paths[i]
        shutil.copy(os.path.join(input_dir, c, img), os.path.join(output_dir, f"fold{fold}", "test", c, img))

IMG_SIZE, BATCH_SIZE = (224, 224), 16
accuracies = []
best_val_accuracy = 0.0
best_fold = -1

all_true_labels = []
all_pred_labels = []

best_history1 = None
best_history2 = None


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30, zoom_range=0.3, horizontal_flip=True, vertical_flip=True,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, brightness_range=[0.8, 1.2]
)
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # used for val AND test

for fold in range(1, k + 1):
    print(f"\n--- Fold {fold} ---")
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    train_generator = train_datagen.flow_from_directory(os.path.join(output_dir, f"fold{fold}/train"),
                                                        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                        class_mode="binary", shuffle=True, seed=42)

    val_generator = eval_datagen.flow_from_directory(os.path.join(output_dir, f"fold{fold}/val"),
                                                     target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                     class_mode="binary", shuffle=False)

    test_generator = eval_datagen.flow_from_directory(os.path.join(output_dir, f"fold{fold}/test"),
                                                      target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                      class_mode="binary", shuffle=False)

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

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    early_stop_p1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # CHANGED: validation_data=val_generator (NOT test_generator)
    history1 = model.fit(train_generator, epochs=25, validation_data=val_generator, callbacks=[early_stop_p1],
                         class_weight=class_weight_dict, verbose=1)

    base_model.trainable = True
    for layer in base_model.layers[:-50]: layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    early_stop_p2 = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    train_generator.reset()

    history2 = model.fit(train_generator, epochs=30, validation_data=val_generator, callbacks=[early_stop_p2],
                         class_weight=class_weight_dict, verbose=1)

    # First and only time test_generator is used for this fold
    loss, accuracy = model.evaluate(test_generator, verbose=0)
    val_loss_final, val_accuracy_final = model.evaluate(val_generator, verbose=0)
    accuracies.append(accuracy)
    print(f"Fold {fold} Test Accuracy : {accuracy * 100:.2f}%  (Val Accuracy: {val_accuracy_final * 100:.2f}%)")

    test_generator.reset()
    y_true_fold = test_generator.classes
    y_pred_prob_fold = model.predict(test_generator)
    y_pred_fold = (y_pred_prob_fold > 0.5).astype("int32").flatten()

    all_true_labels.extend(y_true_fold)
    all_pred_labels.extend(y_pred_fold)

    if val_accuracy_final > best_val_accuracy:
        best_val_accuracy = val_accuracy_final
        best_fold = fold
        best_history1 = history1.history
        best_history2 = history2.history
        model.save("best_directory_kfold.keras")
        print(f"   💾 New best model saved from Fold {fold} (selected by val accuracy)")


print(f"\n{'=' * 40}")
print(f"  FINAL DIRECTORY K-FOLD RESULTS")
print(f"{'=' * 40}")
print(f"🏆 Headline metric — Mean Test Accuracy : {np.mean(accuracies) * 100:.2f}%  (± {np.std(accuracies) * 100:.2f}%)")
print(f"💾 Saved deployment model               : Fold {best_fold} (selected by val accuracy: {best_val_accuracy * 100:.2f}%)")

cm = confusion_matrix(all_true_labels, all_pred_labels)
print("\n📊 Overall Confusion Matrix (All Folds Combined):\n", cm)

precision = precision_score(all_true_labels, all_pred_labels)
recall = recall_score(all_true_labels, all_pred_labels)
f1 = f1_score(all_true_labels, all_pred_labels)

print(f"\nOverall Precision : {precision:.4f}")
print(f"Overall Recall    : {recall:.4f}")
print(f"Overall F1 Score  : {f1:.4f}")
print("\n📋 Overall Classification Report:\n",
      classification_report(all_true_labels, all_pred_labels, target_names=classes,digits=4))

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
print("\n[INFO] Best fold learning c"
      "urves saved as 'kfold_curves.png'")
plt.show()