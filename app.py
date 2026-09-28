import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

st.set_page_config(page_title="VGG16 Image Classifier", page_icon="🖼️")

st.title("Image Classification Using Pre-Trained VGG16")
st.write("Student: PATNAIKUNI CHANDINI | Roll No: 24345A0521")
st.write("Upload an image and VGG16 (ImageNet weights) will predict its Top-5 classes.")


@st.cache_resource
def load_model():
    # Loaded once and reused, so the app does not reload 500+ MB on every click
    return VGG16(weights="imagenet", include_top=True)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the VGG16 model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image (JPG, JPEG or PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("No image uploaded yet. Please choose an image file.")
else:
    try:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    except Exception:
        st.error("This file is not a valid image. Please upload a JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running VGG16 inference..."):
        resized = image.resize((224, 224))
        batch = np.expand_dims(np.array(resized, dtype="float32"), axis=0)
        batch = preprocess_input(batch)
        predictions = model.predict(batch)
        top5 = decode_predictions(predictions, top=5)[0]

    results = pd.DataFrame(
        {
            "Rank": range(1, 6),
            "Class": [label for (_, label, _) in top5],
            "Confidence (%)": [round(float(score) * 100, 2) for (_, _, score) in top5],
        }
    )

    st.subheader("Top-5 Predictions")
    st.table(results.set_index("Rank"))
    st.bar_chart(results.set_index("Class")["Confidence (%)"])
