import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import wandb
from wandb.integration.keras import WandbMetricsLogger

# --- W&B Setup ---
config = {
    "initial_learning_rate": 1e-5,
    "epochs": 50,
    "batch_size": 32,
    "architecture": "ResNet50-FullyFineTuned",
    "image_size": 224
}
run_name = "finetune_resnet_full_advanced"
wandb.init(project="brain-tumor-classification", config=config, name=run_name)

# --- Load Your Best Trained ResNet Model ---
print("Loading the best ResNet model...")
MODEL_PATH = "model_resnet.keras" 
# FIX: We need to tell Keras how to load the custom preprocessing function
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": tf.keras.applications.resnet.preprocess_input
    }
)

# --- Unfreeze the ENTIRE Base Model ---
# Note: The base_model is the 'resnet50' layer inside your Sequential model
base_model = model.get_layer('resnet50') 
base_model.trainable = True
print("The entire ResNet50 base model has been unfrozen for fine-tuning.")

# --- Re-Compile the Model with a Low Learning Rate ---
model.compile(optimizer=Adam(learning_rate=config["initial_learning_rate"]), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# --- Data Loading ---
IMAGE_SIZE = (config["image_size"], config["image_size"])
BATCH_SIZE = config["batch_size"]
BASE_DIR = "data"
TRAIN_DIR = os.path.join(BASE_DIR, "Training")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

# --- Start Advanced Fine-Tuning ---
print(f"--- Starting Advanced Fine-Tuning ---")
history = model.fit(
    train_dataset,
    epochs=config["epochs"],
    validation_data=validation_dataset,
    callbacks=[
        WandbMetricsLogger(log_freq="epoch"),
        ModelCheckpoint(
            filepath=os.path.join(wandb.run.dir, f"{run_name}.keras"),
            monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(
            monitor="val_loss", patience=10, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, verbose=1, min_lr=1e-7)
    ]
)

wandb.finish()