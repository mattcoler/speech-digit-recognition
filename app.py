import streamlit as st

# Page config - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Speech Digit Recognizer", layout="wide")

# Now import everything else
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import traceback
import io
from model import DigitRecognizer
from data_loader import DigitDataset

# Import pydub for audio processing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    st.warning("pydub not available. Audio processing may be limited.")

# Load model function with robust error handling
@st.cache_resource
def load_model(model_path):
    """Load and cache the trained model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DigitRecognizer()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.stop()

# The rest of your code remains the same
# ...
# Convert audio file to compatible format
def convert_audio_to_wav(audio_bytes, output_path):
    """Convert audio to WAV format using pydub"""
    try:
        if not PYDUB_AVAILABLE:
            return False, "pydub not available for audio conversion"
        
        # Create a temporary file to store the original audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_in_file:
            tmp_in_file.write(audio_bytes)
            tmp_in_path = tmp_in_file.name
        
        # Load with pydub and convert to WAV
        try:
            audio = AudioSegment.from_file(tmp_in_path)
            audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
            audio = audio.set_channels(1)  # Convert to mono
            audio.export(output_path, format="wav")
            return True, None
        except Exception as e:
            return False, f"Error converting audio: {str(e)}"
        finally:
            # Clean up temp input file
            if os.path.exists(tmp_in_path):
                os.unlink(tmp_in_path)
    except Exception as e:
        return False, f"Unexpected error in audio conversion: {str(e)}"

# Audio processing function with error handling
def predict_audio(model, audio_path, device):
    """Process audio and make prediction"""
    try:
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
        
        return predicted_idx, confidence, mel_spec[0], all_probs, waveform, sample_rate
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        st.exception(e)
        return None, None, None, None, None, None

# Visualization functions
def create_mel_spectrogram_plot(mel_spec):
    """Create a matplotlib figure for the mel spectrogram"""
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        img = ax.imshow(mel_spec[0].cpu().numpy(), aspect='auto', origin='lower')
        ax.set_title('Mel Spectrogram')
        ax.set_ylabel('Mel Frequency Bands')
        ax.set_xlabel('Time Frames')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        return fig
    except Exception as e:
        st.error(f"Error creating mel spectrogram plot: {str(e)}")
        return None

def create_waveform_plot(waveform, sample_rate):
    """Create a matplotlib figure for the waveform"""
    try:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(waveform[0].numpy())
        ax.set_title('Waveform')
        ax.set_ylabel('Amplitude')
        ax.set_xlabel(f'Sample (at {sample_rate}Hz)')
        return fig
    except Exception as e:
        st.error(f"Error creating waveform plot: {str(e)}")
        return None

def create_probabilities_plot(digits, all_probs, predicted_idx):
    """Create a bar chart of class probabilities"""
    try:
        probs_fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(digits, all_probs)
        ax.set_ylabel('Probability (%)')
        ax.set_title('Class Probabilities')
        
        # Highlight the predicted class
        if predicted_idx is not None and 0 <= predicted_idx < len(bars):
            bars[predicted_idx].set_color('green')
        
        return probs_fig
    except Exception as e:
        st.error(f"Error creating probabilities plot: {str(e)}")
        return None

# Main function with comprehensive error handling
def main():
    st.title("Speech Digit Recognizer")
    st.write("## Recognize spoken digits (0-9) using deep learning")
    st.write("Upload an audio file of a spoken digit or record one directly.")
    
    # Load model
    model_path = "models/Digit Recognizer Best.pth"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.write("Please make sure the model file exists in the correct location.")
        return
    
    # Load the model
    model, device = load_model(model_path)
    st.sidebar.success("Model loaded successfully!")
    
    # Digit labels
    digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Upload Audio", "Record Audio"])
    
    # Upload Audio option
    if input_method == "Upload Audio":
        st.write("### Upload an audio file")
        audio_file = st.file_uploader("Choose a WAV, MP3, or OGG file", type=["wav", "mp3", "ogg"])
        
        if audio_file is not None:
            # Display audio player
            st.audio(audio_file, format="audio/wav")
            
            # Create temp files for processing
            temp_original = None
            temp_converted = None
            
            try:
                # Save uploaded file to a temporary location
                audio_bytes = audio_file.getvalue()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    temp_original = tmp_file.name
                
                # Create another temp file for the converted audio
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_conv_file:
                    temp_converted = tmp_conv_file.name
                
                # Convert audio to a compatible format
                st.write("Processing audio file...")
                success, error_msg = convert_audio_to_wav(audio_bytes, temp_converted)
                
                if not success:
                    # Fallback: try direct processing
                    with open(temp_original, 'wb') as f:
                        f.write(audio_bytes)
                    st.warning(f"Audio conversion failed: {error_msg}. Trying direct processing...")
                    temp_path = temp_original
                else:
                    st.success("Audio converted successfully!")
                    temp_path = temp_converted
                
                # Process and predict
                predicted_idx, confidence, mel_spec, all_probs, waveform, sample_rate = predict_audio(model, temp_path, device)
                
                # Display results if prediction was successful
                if predicted_idx is not None:
                    predicted_digit = digits[predicted_idx]
                    
                    # Show prediction results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Audio Analysis")
                        
                        # Show waveform if available
                        if waveform is not None:
                            st.write("### Waveform")
                            wave_fig = create_waveform_plot(waveform, sample_rate)
                            if wave_fig:
                                st.pyplot(wave_fig)
                        
                        # Show mel spectrogram if available
                        if mel_spec is not None:
                            st.write("### Mel Spectrogram")
                            mel_fig = create_mel_spectrogram_plot(mel_spec)
                            if mel_fig:
                                st.pyplot(mel_fig)
                    
                    with col2:
                        st.subheader("Prediction Results")
                        st.write(f"### Predicted Digit: {predicted_digit.upper()}")
                        st.write(f"Confidence: {confidence:.2f}%")
                        
                        # Show all class probabilities as a bar chart
                        if all_probs is not None:
                            st.write("### All Class Probabilities")
                            probs_fig = create_probabilities_plot(digits, all_probs, predicted_idx)
                            if probs_fig:
                                st.pyplot(probs_fig)
                        
                        # Explanation
                        st.write("### How It Works")
                        st.write("""
                        1. The audio is converted to a mel spectrogram, which represents the sound in a way that matches human hearing perception
                        2. The spectrogram is fed into a convolutional neural network
                        3. The network predicts the probability of each digit class
                        4. The digit with the highest probability is selected as the prediction
                        """)
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.exception(e)
            finally:
                # Clean up temp files
                for temp_file in [temp_original, temp_converted]:
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
    
    # Record Audio option
    elif input_method == "Record Audio":
        st.write("### Record a spoken digit")
        st.write("Click the 'Record' button below and speak a digit (0-9)")
        
        # Wrap the audio recording in try/except to handle environments where it's not supported
        try:
            # Record audio
            audio_recorder = st.audio_recorder(sample_rate=16000)
            
            if audio_recorder is not None:
                audio_bytes = audio_recorder
                
                if audio_bytes:
                    # Display audio player
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Create temp files for processing
                    temp_original = None
                    temp_converted = None
                    
                    try:
                        # Create temp files
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            temp_original = tmp_file.name
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_conv_file:
                            temp_converted = tmp_conv_file.name
                        
                        # Convert audio to a compatible format
                        st.write("Processing audio recording...")
                        success, error_msg = convert_audio_to_wav(audio_bytes, temp_converted)
                        
                        if not success:
                            # Fallback: try direct processing
                            with open(temp_original, 'wb') as f:
                                f.write(audio_bytes)
                            st.warning(f"Audio conversion failed: {error_msg}. Trying direct processing...")
                            temp_path = temp_original
                        else:
                            st.success("Audio converted successfully!")
                            temp_path = temp_converted
                        
                        # Process and predict
                        predicted_idx, confidence, mel_spec, all_probs, waveform, sample_rate = predict_audio(model, temp_path, device)
                        
                        # Display results if prediction was successful
                        if predicted_idx is not None:
                            predicted_digit = digits[predicted_idx]
                            
                            # Show prediction results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Audio Analysis")
                                
                                # Show waveform if available
                                if waveform is not None:
                                    st.write("### Waveform")
                                    wave_fig = create_waveform_plot(waveform, sample_rate)
                                    if wave_fig:
                                        st.pyplot(wave_fig)
                                
                                # Show mel spectrogram if available
                                if mel_spec is not None:
                                    st.write("### Mel Spectrogram")
                                    mel_fig = create_mel_spectrogram_plot(mel_spec)
                                    if mel_fig:
                                        st.pyplot(mel_fig)
                            
                            with col2:
                                st.subheader("Prediction Results")
                                st.write(f"### Predicted Digit: {predicted_digit.upper()}")
                                st.write(f"Confidence: {confidence:.2f}%")
                                
                                # Show all class probabilities as a bar chart
                                if all_probs is not None:
                                    st.write("### All Class Probabilities")
                                    probs_fig = create_probabilities_plot(digits, all_probs, predicted_idx)
                                    if probs_fig:
                                        st.pyplot(probs_fig)
                    except Exception as e:
                        st.error(f"Error processing recorded audio: {str(e)}")
                        st.exception(e)
                    finally:
                        # Clean up temp files
                        for temp_file in [temp_original, temp_converted]:
                            if temp_file and os.path.exists(temp_file):
                                try:
                                    os.unlink(temp_file)
                                except:
                                    pass
        except Exception as e:
            st.error("Audio recording is not available in this environment.")
            st.info("This might be due to browser permissions or Streamlit Cloud limitations. Please try uploading an audio file instead.")
    
    # Add information about the project
    st.sidebar.markdown("---")
    st.sidebar.write("### About")
    st.sidebar.write("""
    This application uses a CNN trained on the SPEECHCOMMANDS dataset to recognize spoken digits.
    
    The audio is processed into mel spectrograms, which are then fed into the neural network for classification.
    """)
    st.sidebar.write("[GitHub Repository](https://github.com/mattcoler/speech-digit-recognition)")
    
    # Add information about mel spectrograms
    st.sidebar.markdown("---")
    st.sidebar.write("### What is a Mel Spectrogram?")
    st.sidebar.write("""
    A mel spectrogram is a visual representation of sound that's specifically designed to match human hearing perception.
    
    Unlike a regular spectrogram that uses a linear frequency scale, the mel scale stretches the lower frequencies (where we can distinguish small changes) and compresses higher frequencies (where we need bigger changes to notice differences).
    
    This makes mel spectrograms perfect for speech recognition because they emphasize the frequencies where human speech carries the most important information.
    """)

    # Add information about project status
    st.sidebar.markdown("---")
    st.sidebar.write("### Project Status")
    st.sidebar.info("""
    This is a demonstration project with a model that achieves 96.82% accuracy on the test set.
    
    Audio processing in web browsers can be challenging due to format compatibility issues. If you encounter problems, try using WAV files recorded at 16kHz, mono.
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error(traceback.format_exc())
