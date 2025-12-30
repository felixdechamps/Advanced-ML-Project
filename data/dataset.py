"""
Dataset loader for PhysioNet/CinC Challenge 2017
Based on Hannun et al. (2019) data processing methodology
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import scipy.io as sio

class PhysioNet2017Dataset(Dataset):
    """
    PhysioNet Challenge 2017 dataset loader
    
    Reference: Clifford et al. (2017) "AF Classification from a Short Single Lead 
    ECG Recording: the PhysioNet/Computing in Cardiology Challenge 2017"
    
    Dataset structure from PhysioNet 2017:
    - 8,528 single channel ECG recordings
    - Sampled at 300 Hz
    - 16-bit resolution
    - Average duration of 30s
    - Classes: Normal (N), AF (A), Other (O), Noisy (~)
    """
    
    def __init__(self, data_dir, references_file='REFERENCE.csv', 
                 transform=None, target_length=9000):
        """
        Args:
            data_dir: Directory containing .mat files
            references_file: CSV file with labels (format: filename,label)
            transform: Optional transform to be applied on ECG signal
            target_length: Target length for ECG signals (default 9000 = 30s at 300Hz)
                          Hannun et al.: "30 second records"
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_length = target_length
        
        # Load references
        # PhysioNet 2017 format: filename,label (e.g., A00001,N)
        ref_path = os.path.join(data_dir, references_file)
        self.references = pd.read_csv(ref_path, names=['filename', 'label'])
        
        # Map labels to integers
        # As per Clifford et al. (2017) and used in Hannun et al. evaluation
        self.label_map = {
            'N': 0,  # Normal sinus rhythm
            'A': 1,  # Atrial Fibrillation
            'O': 2,  # Other rhythms
            '~': 3   # Noisy
        }
        
        self.references['label_id'] = self.references['label'].map(self.label_map)
    
    def __len__(self):
        """Returns the total number of samples"""
        return len(self.references)
    
    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index
        
        Returns:
            signal: ECG signal tensor of shape (target_length,)
            label: Class label (0-3)
        """
        # Get filename and label
        filename = self.references.iloc[idx]['filename']
        label = self.references.iloc[idx]['label_id']
        
        # Load .mat file
        # PhysioNet 2017: "single-channel ECG signals stored in MATLAB v4 format"
        mat_path = os.path.join(self.data_dir, f'{filename}.mat')
        mat_data = sio.loadmat(mat_path)
        
        # Extract signal
        # PhysioNet format: data stored in 'val' field
        signal = mat_data['val'].squeeze().astype(np.float32)
        
        # Preprocessing following Hannun et al. methodology
        # Hannun et al.: "network takes as input only the raw ECG samples"
        # But signals need length standardization
        signal = self._preprocess_signal(signal)
        
        # Apply additional transforms if specified
        if self.transform:
            signal = self.transform(signal)
        
        return torch.from_numpy(signal), label
    
    def _preprocess_signal(self, signal):
        """
        Preprocess ECG signal to target length
        
        Hannun et al. used 30-second segments. For variable length recordings:
        - If longer: truncate to target_length
        - If shorter: zero-pad to target_length
        
        This follows common practice in hsd1503/resnet1d repository
        """
        current_length = len(signal)
        
        if current_length >= self.target_length:
            # Truncate: take first target_length samples
            # Alternative: random crop (not used in Hannun et al. for test set)
            signal = signal[:self.target_length]
        else:
            # Zero-pad to target length
            # Hannun et al. paper doesn't explicitly mention padding strategy
            # but this is standard practice for variable-length sequences
            pad_length = self.target_length - current_length
            signal = np.pad(signal, (0, pad_length), mode='constant', constant_values=0)
        
        # Normalize signal
        # Hannun et al.: "Batch Normalization" is applied in network
        # but input normalization helps training stability (hsd1503 approach)
        # Z-score normalization: (x - mean) / std
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        
        if signal_std > 0:
            signal = (signal - signal_mean) / signal_std
        
        return signal
    
    def get_class_distribution(self):
        """
        Returns class distribution for weighted sampling
        
        Hannun et al.: "rare rhythms... were intentionally oversampled"
        This method helps implement balanced training
        """
        class_counts = self.references['label_id'].value_counts().sort_index()
        return class_counts.values