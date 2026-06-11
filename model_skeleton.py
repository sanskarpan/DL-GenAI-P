import torch
import torch.nn as nn
import glob
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class PrecomputedFeatureDataset(Dataset):
    def __init__(self, features_dir):
        self.files = glob.glob(os.path.join(features_dir, '**', '*.pt'), recursive=True)
        self.genres = sorted(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        # Extract genre from the directory name
        genre = Path(file_path).parent.name
        label = self.genre_to_idx[genre] # converts genre name to a numerically encoded value
        
        # Load precomputed tensor
        feature = torch.load(file_path)
        return feature, label

class CRNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # TODO 1: Define the CNN Backbone using nn.Sequential
        # Expect an input of shape (Batch_Size, 1, 128, Time_Steps)
        # Block 1: Conv2d(1 -> 32, kernel=3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
        # Block 2: Conv2d(32 -> 64, kernel=3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
        self.cnn = None 
        
        # TODO 2: Define the RNN Component
        # Hint: Calculate the flattened feature size coming out of your CNN.
        # Original Mels = 128. After two MaxPool2d(2) layers, Mels = 128 / 4 = 32.
        # Channels = 64. So, Input Size = Channels * Mels = 2048.
        # Create a 1-layer Bidirectional LSTM with hidden_size=64 and batch_first=True.
        self.lstm = None
        
        # TODO 3: Define the Classifier
        # Create a Fully Connected (Linear) layer. 
        # Hint: Since the LSTM is bidirectional, the input features will be hidden_size * 2.
        self.fc = None

    def forward(self, x):
        """
        Input:
            x: Tensor of shape (Batch, 1, 128, Time) representing Mel-spectrograms.
        Output:
            logits: Tensor of shape (Batch, num_classes) representing unnormalized class scores.
        """
        
        # TODO 4: Pass 'x' through the CNN backbone
        # Expected shape after CNN: (Batch, Channels=64, Mels=32, Time)
        
        # TODO 5: Bridge the gap (Shape Manipulation)
        # Permute and reshape the CNN output to be compatible with the LSTM.
        # Extract b, c, f, t from the tensor shape.
        # Permute the dimensions to bring Time forward, then reshape to flatten Channels and Mels.
        # Target shape for LSTM: (Batch, Time_Steps, Channels * Mels)
        
        # TODO 6: Pass the reshaped sequence through the LSTM
        # The LSTM returns output and (hidden_state, cell_state). You only need the output.
        
        # TODO 7: Global Max Pooling over the time dimension
        # Collapse the sequence down to a single vector using torch.max() over the time dimension (dim=1).
        # Note: torch.max returns both values and indices. You only need the values.
        
        # TODO 8: Pass the pooled vector through the fully connected layer
        
        # return logits
        raise NotImplementedError("You need to implement the CRNN forward pass!")