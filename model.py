import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = tf.keras.models.load_model("brain_tumor_model.h5")

def predict_image(image_file):
    image = load_img(image_file, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor"
