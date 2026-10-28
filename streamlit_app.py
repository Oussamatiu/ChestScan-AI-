import streamlit as st
import tensorflow as tf

from src.predict import load_prediction_settings, predict_uploaded_image


st.set_page_config(page_title="ChestScan AI")
st.title("ChestScan AI")
st.warning(
    "Educational prototype only. This model is not a medical diagnosis and "
    "must not replace advice from a qualified healthcare professional."
)


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


try:
    model_path, threshold = load_prediction_settings()
    model = load_model(str(model_path))
except (KeyError, ValueError, OSError) as error:
    st.error(f"Model not ready: {error}")
    st.stop()

image_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["jpg", "jpeg", "png", "bmp"],
)

if image_file is not None:
    st.image(image_file, caption="Uploaded X-ray", use_container_width=True)

    if st.button("Predict"):
        try:
            result = predict_uploaded_image(
                image_file.getvalue(),
                model,
                threshold,
            )
        except (ValueError, OSError) as error:
            st.error(f"Could not read this image: {error}")
        else:
            st.subheader(f"Prediction: {result['prediction']}")
            st.write(
                f"Probability of NORMAL: {result['probability_normal']:.1%}"
            )
            st.write(
                f"Probability of PNEUMONIA: "
                f"{result['probability_pneumonia']:.1%}"
            )
