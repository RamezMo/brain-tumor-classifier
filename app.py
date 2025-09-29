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
def load_best_model():
    """
    Loads your best-performing model from file.
    The @st.cache_resource decorator ensures this is loaded only once.
    """
    # This must point to the .keras file from your best run
    model_path = "finetune_resnet_full_advanced.keras"
    st.write("Loading model...")
    # Load the model with the custom object for the preprocessing function
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input": tf.keras.applications.resnet.preprocess_input
        }
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
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
    st.write("")

    # Make prediction on button click
    if st.button("Predict Tumor Type"):
        with st.spinner("Analyzing the image..."):
            # Prepare the image for the model
            image_resized = image.resize((224, 224))
            image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
            image_array = np.expand_dims(image_array, axis=0)

            # Make a prediction
            prediction = model.predict(image_array)
            
            # Get the top prediction
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence_score = float(np.max(prediction, axis=1)[0])
            
            # Display Result
            st.success(f"**Prediction:** {predicted_class_name}")
            st.info(f"**Confidence:** {confidence_score:.2%}")