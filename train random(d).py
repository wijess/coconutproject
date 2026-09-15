import os
import random
import numpy as np
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ---- Stage 2a: disease-type dataset (3-class, NOT including nutrient_deficiency) ----
dataset_dir = "dataset_clean_disease"
classes = ["healthy", "gray_leaf_blight", "leaf_rot"]   # match your actual folder names exactly
NUM_CLASSES = len(classes)

IMG_SIZE = 224
BATCH_SIZE = 16

image_paths = []
labels = []
for label, class_name in enumerate(classes):
    class_folder = os.path.join(dataset_dir, class_name)
    for img in sorted(os.listdir(class_folder)):
        image_paths.append(os.path.join(class_folder, img))
        labels.append(label)

image_paths = np.array(image_paths)
labels = np.array(labels)

print(f"Total images: {len(image_paths)}")
for i, c in enumerate(classes):
    print(f"  {c}: {np.sum(labels == i)}")

def load_images(paths, img_size):
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        imgs.append(img)
    return np.array(imgs, dtype="float32")

train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.20, stratify=labels, random_state=42
)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels, test_size=0.10, stratify=train_val_labels, random_state=42
)

print(f"\nTrain: {len(train_paths)}  Val: {len(val_paths)}  Test: {len(test_paths)}")

train_images = load_images(train_paths, IMG_SIZE)
val_images = load_images(val_paths, IMG_SIZE)
test_images = load_images(test_paths, IMG_SIZE)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30, zoom_range=0.3, horizontal_flip=True, vertical_flip=True,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
    brightness_range=[0.8, 1.2]
)
val_images_preprocessed = preprocess_input(val_images.copy())
test_images_preprocessed = preprocess_input(test_images.copy())

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass weights: {class_weight_dict}")

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
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop_p1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\nPhase 1: Frozen base training...")
history1 = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
    epochs=25, validation_data=(val_images_preprocessed, val_labels),
    callbacks=[early_stop_p1], class_weight=class_weight_dict, verbose=1
)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop_p2 = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

print("\nPhase 2: Fine-tuning...")
history2 = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
    epochs=30, validation_data=(val_images_preprocessed, val_labels),
    callbacks=[early_stop_p2], class_weight=class_weight_dict, verbose=1
)

loss, accuracy = model.evaluate(test_images_preprocessed, test_labels, verbose=0)
print(f"\nRandom Split Test Accuracy: {accuracy*100:.2f}%")

y_pred_prob = model.predict(test_images_preprocessed)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report:")
print(classification_report(test_labels, y_pred, target_names=classes, digits=4))
print("\nConfusion Matrix:")
print(confusion_matrix(test_labels, y_pred))

model.save("stage2a_random_split_model.keras")
print("\nModel saved: stage2a_random_split_model.keras")
