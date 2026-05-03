import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import json

# ---------------- PATHS ----------------
# Project root (TumorDetect)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_dir = os.path.join(BASE_DIR, "dataset", "training")
test_dir = os.path.join(BASE_DIR, "dataset", "testing")

print("Train path:", train_dir)
print("Test path:", test_dir)

# ---------------- MODEL DIRECTORY ----------------
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- SETTINGS ----------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

# ---------------- DATA AUGMENTATION ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ---------------- SAVE CLASS LABELS ----------------
class_indices = train_data.class_indices
class_labels = {v: k for k, v in class_indices.items()}

with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
    json.dump(class_labels, f)

print("Class mapping saved:", class_labels)

# ---------------- BASE MODEL ----------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# ---------------- MODEL BUILD ----------------
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# ---------------- COMPILE (STAGE 1) ----------------
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- CALLBACKS ----------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_model.keras"),  # ✅ FIXED
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ---------------- TRAIN STAGE 1 ----------------
model.fit(
    train_data,
    validation_data=test_data,
    epochs=10,
    callbacks=[early_stop, checkpoint]
)

# ---------------- FINE TUNING ----------------
base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN STAGE 2 ----------------
model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# ---------------- FINAL SAVE ----------------
model_path = os.path.join(MODEL_DIR, "model.keras")
model.save(model_path)

print("✅ Model saved at:", model_path)

# ---------------- EVALUATION ----------------
loss, acc = model.evaluate(test_data)
print("✅ Test Accuracy:", acc)

print("🎉 Training complete!")