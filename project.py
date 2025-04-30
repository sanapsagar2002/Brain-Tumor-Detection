import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



# Load your trained CNN model
MODEL_PATH = "cnn-parameters-improvement-09-0.92.keras"
model = tf.keras.models.load_model(MODEL_PATH)



def preprocess_image(image):
    """Preprocess the uploaded image to the required input format for the model."""
    img = image.resize((240, 240))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

def main():
    st.title("Brain Tumor Detection")
    st.write("Upload an image to check for brain tumors.")

    # File upload
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make a prediction
        prediction = model.predict(preprocessed_image)
        result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
        #result = "Tumor Detected" if prediction == 1 else "No Tumor Detected"

        # Display the result
        st.subheader("Result")
        st.write(result)

if __name__ == "__main__":
    main()
