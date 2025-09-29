import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50


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
    st.write("Building model architecture...")
    
    # --- Step 1: Rebuild the exact same model architecture ---
    base_model = ResNet50(
        weights=None,  # Do not load pre-trained weights, we will load our own
        include_top=False,
        input_shape=(224, 224, 3),
        name='resnet50'
    )
    base_model.trainable = True # Set to True as we are loading fine-tuned weights

    # Use the same Sequential structure
    model = Sequential([
        Input(shape=(224, 224, 3), name='input_layer'),
        tf.keras.layers.Lambda(tf.keras.applications.resnet.preprocess_input),
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ], name="BrainTumorClassifier_ResNet")

    # --- Step 2: Load only the weights into the architecture ---
    model_path = "best_model.keras"
    st.write(f"Loading weights from {model_path}...")
    model.load_weights(model_path)
    
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
            image_batch = np.expand_dims(image_array, axis=0)

            prediction = model.predict(image_batch)
            
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence_score = float(np.max(prediction, axis=1)[0])
            
            st.success(f"**Prediction:** {predicted_class_name}")
            st.info(f"**Confidence:** {confidence_score:.2%}")