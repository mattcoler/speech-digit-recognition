import streamlit as st
import torch
import os
import traceback
from model import DigitRecognizer

# Simple page setup
st.title("Speech Digit Recognizer")
st.write("## Recognize spoken digits (0-9) using deep learning")
st.write("Upload an audio file of a spoken digit or record one directly.")

# Try to load the model with detailed error reporting
try:
    # Check if model file exists
    model_path = "models/Digit Recognizer Best.pth"
    st.write(f"Checking for model at: {model_path}")
    
    if os.path.exists(model_path):
        st.success(f"✓ Model file found")
        
        # Try to initialize the model
        st.write("Initializing model...")
        model = DigitRecognizer()
        st.success(f"✓ Model structure initialized")
        
        # Try to load the state dict
        st.write("Loading model weights...")
        device = torch.device("cpu")
        
        # Add debugging for the load_state_dict operation
        try:
            state_dict = torch.load(model_path, map_location=device)
            st.write(f"State dict keys: {list(state_dict.keys())[:5]}...")
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            st.success(f"✓ Model weights loaded successfully")
        except Exception as e:
            st.error(f"Error loading model weights: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            st.error(traceback.format_exc())
    else:
        st.error(f"× Model file not found at {model_path}")
        st.write("Checking directory structure:")
        
        # List root directory
        st.write("Files in root directory:")
        st.write(os.listdir("."))
        
        # Check if models directory exists
        if os.path.exists("models"):
            st.write("Files in models directory:")
            st.write(os.listdir("models"))
        else:
            st.error("× models directory does not exist")
except Exception as e:
    st.error(f"Unexpected error: {str(e)}")
    st.error(f"Error type: {type(e).__name__}")
    st.error(traceback.format_exc())

# Basic information
st.sidebar.markdown("---")
st.sidebar.write("### About")
st.sidebar.write("""
This application uses a CNN trained on the SPEECHCOMMANDS dataset to recognize spoken digits.

The audio is processed into mel spectrograms, which are then fed into the neural network for classification.
""")
st.sidebar.write("[GitHub Repository](https://github.com/mattcoler/speech-digit-recognition)")
