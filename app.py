import streamlit as st
import PyPDF2
import tensorflow as tf
import numpy as np

st.title("PROJECT ANALYSER")

# Input fields for project details
A = st.text_input("Enter your project title:")
B = st.text_area("Enter your project description:")
C = st.text_input("Which technologies are used in the project:")
D = st.text_input("Which model or algorithm gives better accuracy according to your project:")
E = st.text_input("What is the accuracy of your project:")
F = st.text_input("What is job description:")

# File uploader for additional content
uploaded_file = st.file_uploader("Upload file", type=["txt", "pdf"])

# Load pre-trained NLP model
model_path = "C:\\Users\\kisho\\Downloads\\Resume_Data.h5"
nlp_model = tf.keras.models.load_model(model_path)

# Submit button
if st.button("SUBMIT"):
    # Process Uploaded File
    file_content = ""
    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1]
        if file_type == "txt":
            file_content = uploaded_file.read().decode("utf-8")
        elif file_type == "pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                file_content += page.extract_text()
        else:
            st.error("Unsupported file format. Please upload a .txt or .pdf file.")
    
    # Combine All Inputs for Prediction
    project_input = f"{A} {B} {C} {D} {E} {F} {file_content[:1000]}"
    
    # Convert input into model-compatible format
    input_data = np.array([project_input])  # Adjust preprocessing as per model's requirements
    
    # Make prediction
    prediction = nlp_model.predict(input_data)
    
    # Display Prediction
    st.subheader("Project Suitability Prediction")
    st.write(f"Predicted Category: {np.argmax(prediction)}")
    
    # Evaluate Project Based on Accuracy
    try:
        accuracy = float(E)
        if accuracy > 85:
            st.write("This project is **best** and **highly suitable** for your resume.")
        elif 70 <= accuracy <= 85:
            st.write("This project is **better** and can be considered for your resume.")
        else:
            st.write("This project is **not convenient** for your resume. Consider improving its accuracy.")
    except ValueError:
        st.error("Please enter a valid numeric value for accuracy.")
