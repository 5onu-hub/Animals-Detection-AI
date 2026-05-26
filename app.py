# app.py

import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# LOAD MODEL
# =========================

model = load_model("animal_model.keras")

# =========================
# CLASS LABELS
# =========================

classes = {
    0: ' Antelope',
    1: ' Badger',
    2: ' Bat',
    3: ' Bear',
    4: ' Bee',
    5: ' Beetle',
    6: ' Bison',
    7: ' Boar',
    8: ' Butterfly',
    9: ' Cat',
    10: ' Caterpillar',
    11: ' Chimpanzee',
    12: ' Cockroach',
    13: ' Cow',
    14: ' Coyote',
    15: ' Crab',
    16: ' Crow',
    17: 'Deer',
    18: ' Dog',
    19: ' Dolphin',
    20: ' Donkey',
    21: ' Dragonfly',
    22: ' Duck',
    23: ' Eagle',
    24: ' Elephant',
    25: ' Flamingo',
    26: ' Fly',
    27: ' Fox',
    28: ' Goat',
    29: ' Goldfish',
    30: ' Goose',
    31: ' Gorilla',
    32: ' Grasshopper',
    33: ' Hamster',
    34: ' Hare',
    35: ' Hedgehog',
    36: ' Hippopotamus',
    37: ' Hornbill',
    38: ' Horse',
    39: ' Hummingbird',
    40: ' Hyena',
    41: ' Jellyfish',
    42: ' Kangaroo',
    43: ' Koala',
    44: ' Ladybugs',
    45: ' Leopard',
    46: ' Lion',
    47: ' Lizard',
    48: ' Lobster',
    49: ' Mosquito',
    50: ' Moth',
    51: ' Mouse',
    52: ' Octopus',
    53: ' Okapi',
    54: ' Orangutan',
    55: ' Otter',
    56: ' Owl',
    57: ' Ox',
    58: ' Oyster',
    59: ' Panda',
    60: ' Parrot',
    61: ' Pelecaniformes',
    62: ' Penguin',
    63: ' Pig',
    64: ' Pigeon',
    65: ' Porcupine',
    66: ' Possum',
    67: ' Raccoon',
    68: ' Rat',
    69: ' Reindeer',
    70: ' Rhinoceros',
    71: ' Sandpiper',
    72: ' Seahorse',
    73: ' Seal',
    74: ' Shark',
    75: ' Sheep',
    76: ' Snake',
    77: ' Sparrow',
    78: ' Squid',
    79: '️ Squirrel',
    80: ' Starfish',
    81: ' Swan',
    82: ' Tiger',
    83: ' Turkey',
    84: ' Turtle',
    85: ' Whale',
    86: ' Wolf',
    87: ' Wombat',
    88: ' Woodpecker',
    89: ' Zebra'
}

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Animal Detection AI",
    page_icon="🐾"
)

# =========================
# UI
# =========================

st.title("🐾 Animal Detection AI")

st.write("Upload an animal image to identify the animal.")

uploaded_file = st.file_uploader(
    "Choose an animal image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================

if uploaded_file is not None:

    # Display image
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Load image
    img = image.load_img(
        uploaded_file,
        target_size=(224, 224)
    )

    # Convert image to array
    img_array = image.img_to_array(img)

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess
    img_array = preprocess_input(img_array)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    result = classes[predicted_class]

    # =========================
    # LOW CONFIDENCE CHECK
    # =========================

    if confidence < 60:

        st.warning("️ Unable to confidently identify animal")

        st.info(
            f"""
Confidence: {confidence:.2f}%

Possible reasons:
- unclear image
- unsupported animal
- unusual angle
"""
        )

    else:

        st.success(f"🐾 Prediction: {result}")

        st.info(f" Confidence: {confidence:.2f}%")