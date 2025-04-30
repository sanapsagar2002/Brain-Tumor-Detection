import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import io
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import pagesizes

# Load your trained CNN model
MODEL_PATH = "cnn-parameters-improvement-09-0.92.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    """Preprocess the uploaded image to the required input format for the model."""
    img = image.resize((240, 240))  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0), img  # Return both processed array & resized image

def wrap_text(canvas, text, x, y, max_width):
    """Wrap text dynamically within a defined width."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    
    words = text.split()
    line = ""
    for word in words:
        test_line = line + word + " "
        if stringWidth(test_line, "Helvetica", 11) < max_width:
            line = test_line
        else:
            canvas.drawString(x, y, line.strip())
            y -= 15  # Move to the next line
            line = word + " "
    
    if line:
        canvas.drawString(x, y, line.strip())
        y -= 15  # Final line spacing
    return y

def create_pdf(image, result):
    """Generate a well-spaced PDF with text wrapping for recommendations."""
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=pagesizes.A4)

    # Page dimensions
    page_width, page_height = pagesizes.A4
    left_margin = 100
    text_width = 400  # Max text width for wrapping

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left_margin, page_height - 100, "Brain Tumor Detection Report")
    
    # Timestamp
    c.setFont("Helvetica", 10)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(left_margin, page_height - 120, f"Date & Time: {timestamp}")

    y_position = page_height - 160  # Adjust initial y-position

    # Diagnosis Result
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, "Diagnosis:")
    c.setFont("Helvetica", 12)
    c.drawString(left_margin, y_position - 20, result)

    y_position -= 60  # More spacing before recommendation

    # Medical Recommendation
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, "Medical Recommendation:")
    
    c.setFont("Helvetica", 11)
    if result == "Tumor Detected":
        recommendation = (
            "⚠️ Urgent: Please consult a medical professional as soon as possible "
            "for further evaluation and necessary treatment. Early diagnosis and "
            "medical intervention are crucial for better outcomes."
        )
    else:
        recommendation = (
            "✅ No tumor detected. However, if you experience any unusual symptoms such as "
            "persistent headaches, vision problems, or dizziness, consult a doctor for reassurance."
        )
    
    y_position = wrap_text(c, recommendation, left_margin, y_position - 25, text_width)

    # Spacing before the image
    y_position -= 40

    # Convert image for ReportLab
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    img_reader = ImageReader(img_buffer)

    # Draw the image
    c.drawImage(img_reader, left_margin, y_position - 200, width=200, height=200)

    # Adjust spacing after image
    y_position -= 250

    # Disclaimer
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y_position, "Disclaimer:")
    c.setFont("Helvetica-Oblique", 10)
    wrap_text(
        c,
        "This report is for preliminary assessment only. Please consult a medical professional for final diagnosis.",
        left_margin,
        y_position - 20,
        text_width
    )

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer  # Return the PDF file buffer

def main():
    st.title("Brain Tumor Detection")
    st.write("Upload an image to check for brain tumors.")

    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        # Preprocess the image
        preprocessed_image, resized_image = preprocess_image(image)

        # Make a prediction
        prediction = model.predict(preprocessed_image)
        result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"

        st.subheader("Result")
        st.write(result)

        # Generate and provide a download button for the PDF
        pdf_file = create_pdf(resized_image, result)
        st.download_button(
            label="Download Report as PDF",
            data=pdf_file,
            file_name="Brain_Tumor_Detection_Report.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
