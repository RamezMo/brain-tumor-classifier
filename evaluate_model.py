import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet import preprocess_input

# --- Constants ---
IMAGE_SIZE = (224, 224)
BASE_DIR = "data"
TEST_DIR = os.path.join(BASE_DIR, "Testing")
# --- MODIFIED: Point to the final advanced fine-tuned model ---
MODEL_PATH = "finetune_resnet_full_advanced.keras"

# --- 1. Load the Trained Model ---
print(f"Loading trained model from: {MODEL_PATH}")
# We need to tell Keras about the custom Lambda layer when loading
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'preprocess_input': preprocess_input}
)

# --- 2. Load and Preprocess the Test Data ---
def load_test_data(test_dir, image_size):
    images = []
    labels = []
    class_names = sorted(os.listdir(test_dir))
    label_map = {name: i for i, name in enumerate(class_names)}

    print(f"Loading test data from: {test_dir}")
    print(f"Classes found: {class_names}")

    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            try:
                img = cv2.imread(image_path)
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label_map[class_name])
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    
    return np.array(images), np.array(labels), class_names

test_images, test_labels, class_names = load_test_data(TEST_DIR, IMAGE_SIZE)

# --- 3. Make Predictions on the Test Data ---
print("Making predictions on the test set...")
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# --- 4. Generate and Print the Classification Report ---
print("\n--- Classification Report ---")
report = classification_report(test_labels, predicted_labels, target_names=class_names)
print(report)

# --- 5. Generate and Display the Confusion Matrix ---
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(test_labels, predicted_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()