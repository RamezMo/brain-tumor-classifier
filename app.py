import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
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
    Rebuilds the model architecture and loads the saved weights.
    """
    model_path = "best_model.keras"
    # Rebuild the exact same model architecture
    base_model = tf.keras.applications.ResNet50(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3),
        name='resnet50'
    )
    base_model.trainable = True

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3), name='input_layer'),
        tf.keras.layers.Lambda(preprocess_input),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')
    ], name="BrainTumorClassifier_ResNet")

    model.load_weights(model_path)
    return model

model = load_best_model()
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- UI Components ---
st.title("Brain Tumor MRI Classification")
st.write(
    "Upload an MRI scan to classify the tumor type using a ResNet50 model "
    "fine-tuned to **97% accuracy** on its test set."
)

# --- NEW: Add a link to download sample test images ---
st.markdown("To test the model's 97% performance, you can download sample images from the original test set [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?select=Testing).")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
    st.write("")

    if st.button("Predict Tumor Type"):
        with st.spinner("Analyzing the image..."):
            image_resized = image.resize((224, 224))
            image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
            image_batch = np.expand_dims(image_array, axis=0)

            prediction = model.predict(image_batch)
            
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence_score = float(np.max(prediction, axis=1)[0])
            
            st.success(f"**Prediction:** {predicted_class_name}")
            st.info(f"**Confidence:** {confidence_score:.2%}")

# --- NEW: Add an expander to explain model limitations ---
with st.expander("Why might the model be less accurate on other images?"):
    st.write("""
        This model was trained to be a "specialist" on a specific dataset of MRI scans. It achieved 97% accuracy on images that have a similar style, brightness, and quality to its training data.
        
        Images from other sources (like a different hospital or MRI machine) might be a different "dialect" that the model is not an expert on. Making a model that can generalize perfectly to *any* image from *any* source is one of the biggest challenges in AI today!
    """)