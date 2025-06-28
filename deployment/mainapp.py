import streamlit as st

def show():
    # Main content
    st.title("Welcome!")

    st.markdown("""
    ### What is this Dashboard?

    This interactive dashboard helps you:
    - Explore traffic sign image data with **interactive visualizations**.
    - Predict traffic sign classes using a trained **deep learning model**.
    - Understand model performance and improve your traffic sign recognition system.

    ---
    ### Background Problems 
    Recognizing traffic signs accurately is essential for developing safe and efficient autonomous driving systems. Misinterpreting signs can lead to accidents or traffic violations.  
    The goal of this project is to build a reliable model that can classify traffic signs from images, and further on, can be used in self-driving technologies, road monitoring, and driver assistance systems.

    ---

    ### Model Overview
    This app uses a **Convolutional Neural Network (CNN)** model trained to classify various traffic signs. The model used Transfer Learning from MobileNetV2

    The model achieved:
    - **F1 Score on Test Set: 0.95**
    - **F1 Score on Training Set: 0.98**

    ---
    
    ### Dataset Used for Training
    The model was trained on the dataset taken from kaggle, [Traffic Sign Dataset - Classification](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification/data)
    
    ---

    ### How to Use
    - Navigate using the **sidebar** to explore classification analysis or make predictions.
    - Upload an image or select a test sample to classify a traffic sign.
    """)