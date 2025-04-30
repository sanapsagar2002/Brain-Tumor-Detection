import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import cv2


# Load trained CNN model
MODEL_PATH = "cnn-parameters-improvement-09-0.92.keras"
model = tf.keras.models.load_model(MODEL_PATH)


# Function to preprocess the image for the model
def preprocess_image(image):
    """Preprocess the uploaded image for CNN input."""
    img = image.resize((240, 240))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension


# Function to detect tumor using improved thresholding (NO Grad-CAM)
def detect_tumor_region(image):
    """Detects a tumor region using adaptive thresholding and contours."""
    # Convert image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding (better for varying brightness)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 3)

    # Apply Morphological Operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    min_contour_area = 500  # Adjust based on image size
    contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    if contours:
        # Get the largest contour (assumed to be the tumor)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Ensure the box is not covering the entire image
        if w < 0.9 * image.width and h < 0.9 * image.height:
            return x, y, x + w, y + h  # (x_min, y_min, x_max, y_max)

    return None  # No tumor found


# Function to draw a bounding box
def draw_bounding_box(image, box):
    """Draws a red rectangle around the detected tumor region."""
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="red", width=3)
    return image


def main():
    st.title("Brain Tumor Detection")
    st.write("Upload an image to check for brain tumors and highlight the tumor region.")

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"

        st.subheader("Result")
        st.write(result)

        # If tumor detected, apply image processing to find region
        if prediction[0][0] > 0.5:
            box = detect_tumor_region(image)

            if box:
                outlined_image = draw_bounding_box(image.copy(), box)
                st.subheader("Tumor Region Highlighted")
                st.image(outlined_image, caption="Detected Tumor", width=300)
            else:
                st.write("No clear tumor region detected.")

if __name__ == "__main__":
    main()
