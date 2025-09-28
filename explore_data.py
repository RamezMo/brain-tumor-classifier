import os
import cv2
import matplotlib.pyplot as plt
import random

def view_sample_images(data_dir):
    """
    Loads and displays one random image from each tumor class.
    """
    plt.figure(figsize=(12, 8))
    
    # Get the class names from the folder names
    tumor_classes = os.listdir(data_dir)
    print(f"Found classes: {tumor_classes}")

    for i, class_name in enumerate(tumor_classes):
        class_path = os.path.join(data_dir, class_name)
        
        # Get a list of all images in the class folder
        image_files = os.listdir(class_path)
        
        # Choose a random image
        random_image_name = random.choice(image_files)
        image_path = os.path.join(class_path, random_image_name)
        
        # Load the image using OpenCV
        img = cv2.imread(image_path)
        # Convert from BGR (OpenCV default) to RGB for Matplotlib
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display the image
        plt.subplot(1, len(tumor_classes), i + 1)
        plt.imshow(img)
        plt.title(f"Class: {class_name}")
        plt.axis('off')
        
    plt.suptitle("Sample Images from Each Class", fontsize=16)
    plt.show()

if __name__ == '__main__':
    # Define the path to your training data
    training_data_directory = os.path.join("data", "Training")
    
    if os.path.exists(training_data_directory):
        view_sample_images(training_data_directory)
    else:
        print(f"Error: Directory not found at {training_data_directory}")
        print("Please make sure you have downloaded and unzipped the data correctly.")