import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

# --- Model Loading ---
@st.cache_resource
def load_my_model():
    """
    Loads the trained model from file.
    The @st.cache_resource decorator ensures the model is loaded only once.
    """
    model_path = "best_model.keras"
    # Note: We don't need custom_objects here because the model structure
    # in the last successful training run doesn't require it.
    model = tf.keras.models.load_model(model_path)
    return model

model = load_my_model()
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- UI Components ---
st.title("Brain Tumor MRI Classification")
st.write("Upload an MRI scan to classify the tumor type using the trained ResNet50 model (97% Accuracy).")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Image Preprocessing ---
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Prepare the image for the model
        image_resized = image.resize((224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
        image_array = np.expand_dims(image_array, axis=0)

        # --- Prediction ---
        prediction = model.predict(image_array)
        
        predicted_class_index = np.argmax(prediction, axis=0)[0]
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence_score = np.max(prediction)
        
        # --- Display Result ---
        st.success(f"Prediction: **{predicted_class_name}**")
        st.info(f"Confidence: **{confidence_score:.2%}**")

    except Exception as e:
        st.error(f"An error occurred: {e}")