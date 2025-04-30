import streamlit as st

def run():
    st.title("About Brain Tumors")

    # Description
    st.header("What is a Brain Tumor?")
    st.write("""
    A brain tumor is an abnormal mass of cells in the brain. Some tumors are **benign (non-cancerous)**, while others are **malignant (cancerous and aggressive)**. 
    Brain tumors can affect brain function by pressing on surrounding tissues, causing neurological symptoms.
    """)

    # Types of Tumors
    st.subheader("Types of Brain Tumors")
    st.write("""
    - **Benign Tumors**: Slow-growing, non-cancerous, and less likely to spread.
    - **Malignant Tumors**: Cancerous, grow quickly, and can spread to other brain areas.
    - **Primary Brain Tumors**: Originate in the brain (e.g., gliomas, meningiomas).
    - **Secondary (Metastatic) Brain Tumors**: Spread from other parts of the body (e.g., lung or breast cancer).
    """)

    # Common Symptoms
    st.header("Common Symptoms of Brain Tumors")
    st.write("""
    Symptoms vary depending on the tumor's location, size, and type. Some of the most common symptoms include:
    
    - **Persistent Headaches** (often worse in the morning)
    - **Seizures** (sudden involuntary movements or convulsions)
    - **Vision Problems** (blurred vision, double vision, or loss of peripheral vision)
    - **Memory Problems & Confusion** (difficulty concentrating, personality changes)
    - **Difficulty Speaking or Understanding Speech**
    - **Loss of Balance & Coordination**
    """)

    # Guide on How to Use the Model
    st.header("How to Use the Brain Tumor Detection Model")
    st.write("""
    Follow these steps to check for a brain tumor using our deep-learning model:

    1. **Navigate to the Home Page** (Use the sidebar to go to "Home").
    2. **Upload an CT-Scan Image** (Click on "Browse Image" and select a valid CT scan).
    3. **Model Prediction** (The model will process the image and predict whether a tumor is present).
    4. **View Results** (The output will be displayed as either "Tumor Detected" or "No Tumor Detected").
    
    > 📌 **Note**: This model provides an AI-based preliminary diagnosis. For an accurate medical assessment, consult a healthcare professional.
    """)

    # Final Note
    st.info("This project aims to assist in early detection using AI. If you experience symptoms, seek medical attention.")

