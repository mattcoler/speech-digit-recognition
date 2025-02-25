import torch
import torchaudio
import matplotlib.pyplot as plt
import argparse
import os

from model import DigitRecognizer
from data_loader import DigitDataset

def load_model(model_path, num_classes=10):
    """Load a trained model from disk"""
    model = DigitRecognizer(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_audio(model, audio_path, device):
    """Predict digit from an audio file"""
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
    
    return predicted_idx, confidence, mel_spec[0]

def main():
    parser = argparse.ArgumentParser(description='Predict digit from audio file')
    parser.add_argument('--model_path', type=str, required=True, help='path to model file')
    parser.add_argument('--audio_path', type=str, required=True, help='path to audio file')
    args = parser.parse_args()
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = load_model(args.model_path)
    model = model.to(device)
    
    # Make prediction
    digit_idx, confidence, mel_spec = predict_audio(model, args.audio_path, device)
    
    # Map index to digit name
    digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    predicted_digit = digits[digit_idx]
    
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Visualize the mel spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec[0].cpu().numpy(), aspect='auto', origin='lower')
    plt.title(f'Mel Spectrogram - Predicted: {predicted_digit}')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
