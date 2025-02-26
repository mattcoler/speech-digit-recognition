import streamlit as st
import torch
import torchaudio
import matplotlib.pyplot as plt
import tempfile
import os
from model import DigitRecognizer
from data_loader import DigitDataset

# Page config
st.set_page_config(page_title="Speech Digit Recognizer", layout="wide")

# Load model function
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitRecognizer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Main app content
st.title("Speech Digit Recognizer")
st.write("## Recognize spoken digits (0-9) using deep learning")
st.write("Upload an audio file of a spoken digit or record one directly.")

# Load model
model_path = "models/Digit Recognizer Best.pth"
try:
    model, device = load_model(model_path)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {str(e)}")
    st.stop()

# Digit labels
digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# Input method
input_method = st.radio("Choose input method:", ["Upload Audio", "Record Audio"])

# Basic information
st.sidebar.markdown("---")
st.sidebar.write("### About")
st.sidebar.write("""
This application uses a CNN trained on the SPEECHCOMMANDS dataset to recognize spoken digits.

The audio is processed into mel spectrograms, which are then fed into the neural network for classification.
""")
st.sidebar.write("[GitHub Repository](https://github.com/mattcoler/speech-digit-recognition)")
