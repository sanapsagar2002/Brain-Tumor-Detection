import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from PIL import Image

# Load your trained CNN model
MODEL_PATH = "cnn-parameters-improvement-09-0.92.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    """Preprocess the uploaded image to the required input format for the model."""
    img = image.resize((240, 240))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

def make_gradcam_heatmap(img_array, model, layer_name):
    """Generates a Grad-CAM heatmap for the given image and model layer."""
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def overlay_heatmap(image, heatmap):
    """Applies heatmap overlay on the image."""
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_array = np.array(image)
    superimposed_img = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(superimposed_img)

def main():
    st.title("Brain Tumor Detection with Grad-CAM")
    st.write("Upload an image to check for brain tumors.")

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"

        st.subheader("Result")
        st.write(result)

        if prediction[0][0] > 0.5:
            st.subheader("Grad-CAM Visualization")
            layer_name = "conv0"  # Change this to test different layers
            heatmap = make_gradcam_heatmap(preprocessed_image, model, layer_name)
            heatmap_image = overlay_heatmap(image, heatmap)
            st.image(heatmap_image, caption="Tumor Region Highlighted")

if __name__ == "__main__":
    main()