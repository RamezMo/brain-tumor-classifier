import tensorflow as tf
import numpy as np
from PIL import Image
import io
# --- NEW: Import render_template ---
from flask import Flask, request, jsonify, render_template

# --- 1. Initialize the Flask App ---
app = Flask(__name__)

# --- 2. Load the Trained Model ---
MODEL_PATH = "best_model.keras"
print(f"Loading trained model from: {MODEL_PATH}")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": tf.keras.applications.resnet.preprocess_input
    }
)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
print("Model loaded successfully.")

# --- 3. Create a Prediction Function ---
def prepare_image(image):
    image = image.resize((224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# --- NEW: Define the route for the main homepage ---
@app.route("/")
def home():
    # This tells Flask to find and send the index.html file
    return render_template("index.html")

# --- 4. Define the API Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            prepared_image = prepare_image(image)
            prediction = model.predict(prepared_image)
            
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence_score = float(np.max(prediction, axis=1)[0])
            
            return jsonify({
                "prediction": predicted_class_name,
                "confidence": f"{confidence_score:.2%}"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "An unknown error occurred"}), 500

# --- 5. Run the App ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)