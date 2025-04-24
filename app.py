import streamlit as st
from model import predict_image
from PIL import Image
import tempfile

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection")
st.markdown("Upload a brain MRI image to detect tumor presence.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        img.save(temp.name)
        result = predict_image(temp.name)

    st.subheader("Prediction:")
    st.success(result)
