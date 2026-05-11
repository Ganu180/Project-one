import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("mask_detector.h5")

st.title("Face Mask Detection System")
st.write("Upload an image to detect Mask or No Mask")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]

    mask = prediction[0]
    no_mask = prediction[1]

    if mask > no_mask:
        st.success(f"Result: Mask detected ({mask * 100:.2f}%)")
    else:
        st.error(f"Result: No Mask detected ({no_mask * 100:.2f}%)")