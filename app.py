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

    confidence = prediction[0][0]

    if confidence > 0.95:
        return "Prediction: It's a Dog 🐶"
    elif confidence < 0.05:
        return "Prediction: It's a Cat 🐱"
    else:
        return "❌ This image doesn't appear to be a cat or a dog."



interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="AI-Powered Cat vs Dog Classifier",
    description="Upload an image to find out if it's a Cat or Dog!",
)

interface.launch()
