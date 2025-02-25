import streamlit as st
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import os
from model import DigitRecognizer
from data_loader import DigitDataset

# Page config
st.set_page_config(page_title="Speech Digit Recognizer", layout="wide")

@st.cache_resource
def load_model(model_path):
    """Load and cache the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitRecognizer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def predict_audio(model, audio_path, device):
    """Process audio and make prediction"""
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # Create dataset instance to use its preprocessing
    dataset = DigitDataset()
    
    # Convert to mel spectrogram
    mel_spec = dataset.mel_transform(waveform)
    # Convert to log scale and normalize
    mel_spec = torch.log(mel_spec + 1e-9)
    # Ensure consistent size
    mel_spec = dataset._pad_or_truncate(mel_spec)
    
    # Add batch dimension
    mel_spec = mel_spec.unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(mel_spec)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_idx = output.argmax(1).item()
        confidence = probabilities[0][predicted_idx].item() * 100
    
    # Get all class probabilities
    all_probs = probabilities[0].cpu().numpy() * 100
    
    return predicted_idx, confidence, mel_spec[0], all_probs

def create_mel_spectrogram_plot(mel_spec):
    """Create a matplotlib figure for the mel spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(mel_spec[0].cpu().numpy(), aspect='auto', origin='lower')
    ax.set_title('Mel Spectrogram')
    ax.set_ylabel('Mel Frequency Bands')
    ax.set_xlabel('Time Frames')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    return fig

def create_waveform_plot(waveform, sample_rate):
    """Create a matplotlib figure for the waveform"""
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(waveform[0].numpy())
    ax.set_title('Waveform')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel(f'Sample (at {sample_rate}Hz)')
    return fig

def main():
    st.title("Speech Digit Recognizer")
    st.write("""
    ## Recognize spoken digits (0-9) using deep learning
    Upload an audio file of a spoken digit or record one directly.
    """)
    
    # Sidebar - model path
model_path = st.sidebar.text_input(
    "Model Path", 
    "models/Digit Recognizer Best.pth",
    help="Path to the trained model file"
)
    # Load model
    try:
        model, device = load_model(model_path)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Digit labels
    digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Upload Audio", "Record Audio"])
    
    if input_method == "Upload Audio":
        audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
        
        if audio_file is not None:
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.getvalue())
                temp_path = tmp_file.name
            
            # Display audio player
            st.audio(audio_file, format="audio/wav")
            
            # Load audio to get waveform for display
            waveform, sample_rate = torchaudio.load(temp_path)
            
            # Process and predict
            predicted_idx, confidence, mel_spec, all_probs = predict_audio(model, temp_path, device)
            predicted_digit = digits[predicted_idx]
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Show prediction results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Audio Analysis")
                
                # Show waveform
                st.write("### Waveform")
                wave_fig = create_waveform_plot(waveform, sample_rate)
                st.pyplot(wave_fig)
                
                # Show mel spectrogram
                st.write("### Mel Spectrogram")
                mel_fig = create_mel_spectrogram_plot(mel_spec)
                st.pyplot(mel_fig)
            
            with col2:
                st.subheader("Prediction Results")
                st.write(f"### Predicted Digit: {predicted_digit.upper()}")
                st.write(f"Confidence: {confidence:.2f}%")
                
                # Show all class probabilities as a bar chart
                st.write("### All Class Probabilities")
                probs_fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(digits, all_probs)
                ax.set_ylabel('Probability (%)')
                ax.set_title('Class Probabilities')
                
                # Highlight the predicted class
                bars[predicted_idx].set_color('green')
                
                st.pyplot(probs_fig)
                
                # Explanation
                st.write("### How It Works")
                st.write("""
                1. The audio is converted to a mel spectrogram, which represents the sound in a way that matches human hearing perception
                2. The spectrogram is fed into a convolutional neural network
                3. The network predicts the probability of each digit class
                4. The digit with the highest probability is selected as the prediction
                """)
    
    else:  # Record Audio
        st.write("### Record a spoken digit")
        st.write("Click the 'Record' button below and speak a digit (0-9)")
        
        # Record audio
        audio_bytes = st.audio_recorder(sample_rate=16000)
        
        if audio_bytes:
            # Save the recorded audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                temp_path = tmp_file.name
            
            # Display audio player
            st.audio(audio_bytes, format="audio/wav")
            
            # Load audio to get waveform for display
            waveform, sample_rate = torchaudio.load(temp_path)
            
            # Process and predict
            predicted_idx, confidence, mel_spec, all_probs = predict_audio(model, temp_path, device)
            predicted_digit = digits[predicted_idx]
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Show prediction results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Audio Analysis")
                
                # Show waveform
                st.write("### Waveform")
                wave_fig = create_waveform_plot(waveform, sample_rate)
                st.pyplot(wave_fig)
                
                # Show mel spectrogram
                st.write("### Mel Spectrogram")
                mel_fig = create_mel_spectrogram_plot(mel_spec)
                st.pyplot(mel_fig)
            
            with col2:
                st.subheader("Prediction Results")
                st.write(f"### Predicted Digit: {predicted_digit.upper()}")
                st.write(f"Confidence: {confidence:.2f}%")
                
                # Show all class probabilities as a bar chart
                st.write("### All Class Probabilities")
                probs_fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(digits, all_probs)
                ax.set_ylabel('Probability (%)')
                ax.set_title('Class Probabilities')
                
                # Highlight the predicted class
                bars[predicted_idx].set_color('green')
                
                st.pyplot(probs_fig)
    
    # Add information about the project
    st.sidebar.markdown("---")
    st.sidebar.write("### About")
    st.sidebar.write("""
    This application uses a CNN trained on the SPEECHCOMMANDS dataset to recognize spoken digits.
    
    The audio is processed into mel spectrograms, which are then fed into the neural network for classification.
    """)
    st.sidebar.write("[GitHub Repository](https://github.com/mattcoler/speech-digit-recognition)")

if __name__ == "__main__":
    main()
