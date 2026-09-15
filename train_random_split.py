import os
import random
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

input_dir = "dataset_clean"
output_dir = "random_split"
classes = ["coconut", "non_coconut"]

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

all_images = []
for c in classes:
    for img in sorted(os.listdir(os.path.join(input_dir, c))):
        all_images.append((img, c))

train_val_data, test_data = train_test_split(
    all_images, test_size=0.2, random_state=40
)

train_data, val_data = train_test_split(
    train_val_data, test_size=0.1, random_state=40
)

for folder in ["train", "val", "test"]:
    for c in classes:
        os.makedirs(os.path.join(output_dir, folder, c), exist_ok=True)

for img, c in train_data:
    shutil.copy(os.path.join(input_dir, c, img), os.path.join(output_dir, "train", c, img))
for img, c in val_data:
    shutil.copy(os.path.join(input_dir, c, img), os.path.join(output_dir, "val", c, img))
for img, c in test_data:
    shutil.copy(os.path.join(input_dir, c, img), os.path.join(output_dir, "test", c, img))

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30, zoom_range=0.3, horizontal_flip=True, vertical_flip=True,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, brightness_range=[0.8, 1.2]
)

eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

IMG_SIZE = 224
BATCH_SIZE = 16

train_generator = train_datagen.flow_from_directory(
    os.path.join(output_dir, "train"), target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='binary'
)
val_generator = eval_datagen.flow_from_directory(
    os.path.join(output_dir, "val"), target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='binary',
    shuffle=False
)
test_generator = eval_datagen.flow_from_directory(
    os.path.join(output_dir, "test"), target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='binary',
    shuffle=False
)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_generator.classes),
                                     y=train_generator.classes)
class_weight_dict = dict(enumerate(class_weights))

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = Sequential([
    base_model, GlobalAveragePooling2D(), BatchNormalization(),
    Dense(256, activation='relu'), Dropout(0.5),
    Dense(128, activation='relu'), Dropout(0.3),
    Dense(1, activation='sigmoid')
])

print("\n--- Starting Phase 1 (Frozen Base) ---")
model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
early_stop_p1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# CHANGED #4: validation_data=val_generator (NOT test_generator)
history1 = model.fit(train_generator, epochs=25, validation_data=val_generator, callbacks=[early_stop_p1],
                     class_weight=class_weight_dict)

print("\n--- Starting Phase 2 (Fine-Tuning Last 50 Layers) ---")
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
early_stop_p2 = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
# CHANGED #4 (cont.): validation_data=val_generator (NOT test_generator)
history2 = model.fit(train_generator, epochs=30, validation_data=val_generator, callbacks=[early_stop_p2],
                     class_weight=class_weight_dict)

acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

phase1_epochs = len(history1.history['accuracy'])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy', color='blue')
plt.plot(val_acc, label='Validation Accuracy', color='orange')
plt.axvline(x=phase1_epochs - 1, color='red', linestyle='--', label='Phase 2 Transition')
plt.title('Random Split: Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='orange')
plt.axvline(x=phase1_epochs - 1, color='red', linestyle='--', label='Phase 2 Transition')
plt.title('Random Split: Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('random_split_learning_curves.png', dpi=300)
print("\n[INFO] Learning curves saved as 'random_curves.png'")
plt.show()

print("\n--- Evaluating Model on Held-Out Test Set ---")
loss_eval, accuracy_eval = model.evaluate(test_generator)
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Random Split Test Accuracy: {accuracy_eval * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=classes,digits= 4))
