import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

import os
import gdown

model_path = "brain_tumor_model.h5"

if not os.path.exists(model_path):
    # Google Drive file ID extracted from your link
    file_id = "10jH_md02ZFcekE4xCfV2DbLJ2m22Bu2Z"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)
    
# Load the saved model
model =load_model(model_path)

def predict_image(image_file):
    image = load_img(image_file, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor"
