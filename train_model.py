import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- 1. APNE DATASET KA PATH YAHAN DAALEIN ---
# Aapka dataset "Plant_Disease_Dataset" folder ke andar hai
train_dir = 'Plant_Disease_Dataset/train'
valid_dir = 'Plant_Disease_Dataset/valid'
test_dir = 'Plant_Disease_Dataset/test'

# --- 2. MODEL PARAMETERS ---
# Aapki app.py file 160x160 size use kar rahi hai, isliye hum yahan bhi wahi size rakhenge.
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10 # Aap ise baad mein badha sakte hain (e.g., 15-20) for better accuracy

# --- 3. DATA LOAD KARNA ---
print("Loading data...")
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

valid_data = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names
num_classes = len(class_names)
print(f"Total classes found: {num_classes}")
print(class_names) # Yeh aapke sabhi folder names (diseases) ko print karega

# --- 4. MODEL BANANA (TRANSFER LEARNING) ---
print("Building model...")
# Hum ek pehle se bana hua model (MobileNetV2) istemal karenge
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False # Iske weights ko freeze kar denge

# Hum iske upar apni layers jodenge
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# --- 5. MODEL COMPILE KARNA ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary() # Model ka structure dekhein

# --- 6. MODEL TRAIN KARNA ---
print("\n--- Starting Training ---")
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=valid_data
)
print("--- Training Finished ---\n")

# --- 7. MODEL KO TEST DATA PAR EVALUATE KARNA ---
print("--- Evaluating on Test Data ---")
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)
loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --- 8. TRAINED MODEL KO SAVE KARNA ---
# Yeh sabse zaroori step hai!
model.save("models/my_new_plant_disease_model.keras")
print("Model saved to 'models/my_new_plant_disease_model.keras'")