import streamlit as st
import torch
from model import DigitRecognizer

# Simple page setup
st.title("Speech Digit Recognizer")
st.write("This is a simplified test app to diagnose deployment issues.")

# Try to load the model
try:
    st.write("Attempting to load model...")
    # Check if model file exists
    import os
    model_path = "models/Digit Recognizer Best.pth"
    if os.path.exists(model_path):
        st.success(f"Model file found at {model_path}")
        
        # Try to initialize the model
        model = DigitRecognizer()
        st.success("Model initialized successfully")
        
        # Try to load the state dict
        device = torch.device("cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        st.success("Model loaded successfully!")
    else:
        st.error(f"Model file not found at {model_path}")
        st.write("Files in models directory:")
        if os.path.exists("models"):
            st.write(os.listdir("models"))
        else:
            st.error("models directory does not exist")
            
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write(f"Exception type: {type(e).__name__}")
    
# Show some basic information
st.write("## Repository Information")
st.write("This app is part of a speech digit recognition project.")
