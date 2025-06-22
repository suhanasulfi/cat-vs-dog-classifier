import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved trained model
model = load_model("model.h5")

def predict_image(img):
    img = img.convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Prediction: It's a Dog 🐶"
    else:
        return "Prediction: It's a Cat 🐱"

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="AI-Powered Cat vs Dog Classifier",
    description="Upload an image to find out if it's a Cat or Dog!",
)

interface.launch()