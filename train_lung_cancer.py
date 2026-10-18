
from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "datasets" / "lung_cancer"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 15

# Find the directory containing all three class folders
class_names = ["Bengin cases", "Malignant cases", "Normal cases"]
class_dirs = {}

for name in class_names:
    matches = [
        p for p in DATASET_DIR.rglob(name)
        if p.is_dir()
    ]
    if not matches:
        raise FileNotFoundError(f"Cannot find class folder: {name}")
    class_dirs[name] = matches[0]

dataset_root = class_dirs[class_names[0]].parent

for name in class_names:
    if class_dirs[name].parent != dataset_root:
        raise ValueError("Class folders are not in the same directory.")

print("Dataset folder:", dataset_root)

for name in class_names:
    count = sum(
        1 for p in class_dirs[name].rglob("*")
        if p.is_file() and p.suffix.lower() in
        {".jpg", ".jpeg", ".png", ".bmp"}
    )
    print(f"{name}: {count} images")

# Load training and validation datasets
train_ds = keras.utils.image_dataset_from_directory(
    dataset_root,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    dataset_root,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

print("Class mapping:", dict(enumerate(train_ds.class_names)))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# MobileNetV2 transfer learning
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save to a NEW file so the existing model remains untouched
model_path = MODEL_DIR / "lung_cancer_mobilenetv2_retrained.keras"
model.save(model_path)

mapping_path = MODEL_DIR / "lung_cancer_class_names.json"
with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

print("\nTraining completed.")
print("Model saved to:", model_path)
print("Class mapping saved to:", mapping_path)
print("Best validation accuracy:",
      round(max(history.history["val_accuracy"]) * 100, 2), "%")
