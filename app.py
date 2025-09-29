import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
# Import the specific preprocessing function
from tensorflow.keras.applications.resnet import preprocess_input

# --- Page Configuration ---
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

# --- Model Loading ---
@st.cache_resource
def load_best_model():
    """
    Loads your best-performing model from file.
    """
    model_path = "finetune_resnet_full_advanced.keras"
    st.write("Loading model...")
    # --- FIX: Add custom_objects to handle the Lambda layer ---
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'preprocess_input': preprocess_input}
    )
    st.write("Model loaded successfully!")
    return model

model = load_best_model()
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- UI Components ---
st.title("Brain Tumor MRI Classification")
st.write(
    "Upload an MRI scan to classify the tumor type using a ResNet50 model "
    "fine-tuned to **97% accuracy**."
)

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
    st.write("")

    if st.button("Predict Tumor Type"):
        with st.spinner("Analyzing the image..."):
            image_resized = image.resize((224, 224))
            image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
            image_array = np.expand_dims(image_array, axis=0)

            prediction = model.predict(image_array)
            
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence_score = float(np.max(prediction, axis=1)[0])
            
            st.success(f"**Prediction:** {predicted_class_name}")
            st.info(f"**Confidence:** {confidence_score:.2%}")