import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

class DigitDataset(Dataset):
    def __init__(self, root="./", subset="training"):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root,
            download=True,
            subset=subset
        )
        
        # Parameters for feature extraction
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,  # SPEECHCOMMANDS sample rate
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        
        # Fixed length for all spectrograms (time dimension)
        self.fixed_length = 32
        
        # Get only digit samples
        self.digits = ['zero', 'one', 'two', 'three', 'four', 
                      'five', 'six', 'seven', 'eight', 'nine']
        self.label_to_idx = {label: idx for idx, label in enumerate(self.digits)}
        self.samples = self._get_digit_samples()
        
    def _get_digit_samples(self):
        samples = []
        for i in range(len(self.dataset)):
            waveform, _, label, *_ = self.dataset[i]
            if label in self.digits:
                samples.append((waveform, label))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _pad_or_truncate(self, mel_spec):
        """Ensure all spectrograms have the same length."""
        n_channels, n_mels, n_steps = mel_spec.shape
        
        if n_steps < self.fixed_length:
            # Pad if too short
            padding = torch.zeros(n_channels, n_mels, self.fixed_length - n_steps)
            mel_spec = torch.cat((mel_spec, padding), dim=2)
        elif n_steps > self.fixed_length:
            # Truncate if too long
            mel_spec = mel_spec[:, :, :self.fixed_length]
            
        return mel_spec
    
    def __getitem__(self, idx):
        waveform, label = self.samples[idx]
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        # Convert to log scale and normalize
        mel_spec = torch.log(mel_spec + 1e-9)
        # Ensure consistent size
        mel_spec = self._pad_or_truncate(mel_spec)
        
        # Convert label to index
        label_idx = self.label_to_idx[label]
        
        return mel_spec, label_idx

def get_dataloaders(batch_size=32, root="./"):
    """Create and return dataloaders for training and validation"""
    # Create datasets
    train_dataset = DigitDataset(root=root, subset="training")
    valid_dataset = DigitDataset(root=root, subset="validation")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    return train_loader, valid_loader, train_dataset.digits
