# train.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

# =========================
# IMAGE PREPROCESSING
# =========================

train_data = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_data.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    subset="training"
)

val_generator = train_data.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    subset="validation"
)

# =========================
# PRINT CLASS LABELS
# =========================

print("\n🐾 Animal Classes:\n")
print(train_generator.class_indices)

# =========================
# LOAD PRETRAINED MODEL
# =========================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# =========================
# BUILD MODEL
# =========================

model = Sequential([

    Input(shape=(224, 224, 3)),

    base_model,

    GlobalAveragePooling2D(),

    Dense(256, activation="relu"),

    Dropout(0.5),

    Dense(train_generator.num_classes, activation="softmax")
])

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN MODEL
# =========================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# =========================
# SAVE MODEL
# =========================

model.save("animal_model.keras")

print("\n Animal Detection Model Saved Successfully")