"""
1D ResNet architecture for ECG classification
Based on Hannun et al. (2019) and adapted from hsd1503/resnet1d

Key architecture details from Hannun et al. Extended Data Figure 1:
- 34 layers total (33 convolutional + 1 linear output)
- 16 residual blocks with 2 convolutional layers per block
- Shortcut connections (ResNet architecture by He et al., 2016)
- Filter width: 16
- Filters: 32*2^k where k increments every 4th residual block
- Subsampling by factor of 2 every alternate residual block
- Batch Normalization + ReLU (pre-activation design by He et al., 2016)
- Dropout with p=0.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv1dPadSame(nn.Module):
    """
    Custom 1D convolution with 'same' padding
    
    From hsd1503/resnet1d: extends nn.Conv1d to support 'same' padding
    which maintains input length after convolution
    
    Hannun et al. use filter width of 16, requiring proper padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # Standard Conv1d layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation
        )
    
    def forward(self, x):
        """
        Apply convolution with 'same' padding
        Padding calculation ensures output length = input length / stride
        """
        # Calculate required padding
        # From hsd1503/resnet1d padding logic
        net = x
        
        # Compute padding needed for 'same' convolution
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        
        # Apply padding
        net = F.pad(net, (pad_left, pad_right), mode='constant', value=0)
        
        # Apply convolution
        net = self.conv(net)
        
        return net

class MyMaxPool1dPadSame(nn.Module):
    """
    Custom 1D max pooling with 'same' padding
    From hsd1503/resnet1d repository
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)
    
    def forward(self, x):
        net = x
        
        # Compute padding
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        
        # Apply padding and pooling
        net = F.pad(net, (pad_left, pad_right), mode='constant', value=0)
        net = self.max_pool(net)
        
        return net

class BasicBlock1d(nn.Module):
    """
    Basic residual block for 1D signals
    
    Architecture from Hannun et al.:
    - Two convolutional layers per block
    - Pre-activation design: BN -> ReLU -> Conv (He et al., 2016)
    - Dropout between conv layers (p=0.2)
    - Shortcut connection (identity or projection)
    
    Based on hsd1503/resnet1d BasicBlock implementation
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 downsample=None, dropout_rate=0.2):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size (16 in Hannun et al.)
            stride: Stride for convolution (1 or 2)
            downsample: Downsample layer for shortcut if needed
            dropout_rate: Dropout probability (0.2 in Hannun et al.)
        """
        super(BasicBlock1d, self).__init__()
        
        # Hannun et al.: "applied Batch Normalization and rectified linear activation"
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        # First convolution
        # Hannun et al.: "convolutional layers have a filter width of 16"
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
        
        # Second batch norm and activation
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        # Second convolution
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1
        )
        
        # Shortcut connection
        # Hannun et al.: "employed shortcut connections in manner similar to 
        # Residual Network architecture" (He et al., 2016)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        """
        Forward pass with residual connection
        
        Hannun et al. pre-activation design:
        out = x + F(x) where F is the residual function
        """
        identity = x
        
        # Pre-activation + First conv
        # Hannun et al.: "Before each convolutional layer we applied 
        # Batch Normalization and a rectified linear activation"
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv1(out)
        
        # Pre-activation + Second conv
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.conv2(out)
        
        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual
        out += identity
        
        return out

class ResNet1d(nn.Module):
    """
    1D ResNet for ECG classification
    
    Architecture from Hannun et al. (2019) Extended Data Figure 1:
    - Input: Raw ECG (9000 samples for 30s at 300Hz)
    - 16 residual blocks (each with 2 conv layers)
    - Output: 4 classes for PhysioNet 2017
    
    Filter progression: 32*2^k where k increments every 4 blocks
    - Blocks 1-4: 32 filters
    - Blocks 5-8: 64 filters
    - Blocks 9-12: 128 filters
    - Blocks 13-16: 256 filters
    
    Subsampling: Every alternate block (stride=2)
    """
    
    def __init__(self, in_channels=1, base_filters=32, kernel_size=16, 
                 stride=2, n_classes=4, dropout_rate=0.2):
        """
        Args:
            in_channels: Number of input channels (1 for single-lead ECG)
            base_filters: Base number of filters (32 in Hannun et al.)
            kernel_size: Convolution kernel size (16 in Hannun et al.)
            stride: Stride for downsampling (2 in Hannun et al.)
            n_classes: Number of output classes (4 for PhysioNet 2017)
            dropout_rate: Dropout probability (0.2 in Hannun et al.)
        """
        super(ResNet1d, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.in_channels = base_filters
        
        # Hannun et al.: "first... layers of network are special-cased 
        # due to pre-activation block structure"
        # Initial convolution (not explicitly detailed in paper)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=self.kernel_size,
            stride=2
        )
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        
        # Hannun et al.: "16 residual blocks"
        # "32*2^k filters, where k... incremented by one every fourth residual block"
        # "every alternate residual block subsamples its inputs by factor of two"
        
        # Blocks 1-4: 32 filters
        self.layer1 = self._make_layer(
            block=BasicBlock1d,
            out_channels=base_filters,
            blocks=4,
            stride=1  # First block no downsampling
        )
        
        # Blocks 5-8: 64 filters (k=1, so 32*2^1=64)
        self.layer2 = self._make_layer(
            block=BasicBlock1d,
            out_channels=base_filters * 2,
            blocks=4,
            stride=2  # Downsample
        )
        
        # Blocks 9-12: 128 filters (k=2, so 32*2^2=128)
        self.layer3 = self._make_layer(
            block=BasicBlock1d,
            out_channels=base_filters * 4,
            blocks=4,
            stride=2  # Downsample
        )
        
        # Blocks 13-16: 256 filters (k=3, so 32*2^3=256)
        self.layer4 = self._make_layer(
            block=BasicBlock1d,
            out_channels=base_filters * 8,
            blocks=4,
            stride=2  # Downsample
        )
        
        # Global average pooling
        # Not explicitly mentioned in Hannun et al. but standard practice
        # before final linear layer (from hsd1503/resnet1d)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Hannun et al.: "final fully-connected softmax layer produces 
        # distribution over the 12 output classes"
        # For PhysioNet 2017: 4 classes
        self.fc = nn.Linear(base_filters * 8, n_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride):
        """
        Create a layer with multiple residual blocks
        
        Args:
            block: BasicBlock1d class
            out_channels: Number of output channels
            blocks: Number of residual blocks in this layer
            stride: Stride for first block (1 or 2)
        
        Hannun et al.: "every alternate residual block subsamples"
        Implementation: first block of each layer may downsample
        """
        downsample = None
        
        # Create downsample layer if needed for shortcut
        # Required when stride != 1 or channels change
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        
        # First block (may downsample)
        layers.append(
            block(
                in_channels=self.in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                downsample=downsample,
                dropout_rate=self.dropout_rate
            )
        )
        
        self.in_channels = out_channels
        
        # Remaining blocks in this layer
        # Hannun et al.: "every alternate residual block subsamples"
        # Within a layer group, alternate blocks downsample
        for i in range(1, blocks):
            # Determine if this block should downsample
            # Pattern: no downsample, downsample, no downsample, downsample...
            block_stride = 2 if i % 2 == 1 else 1
            
            # Create downsample for this block if needed
            block_downsample = None
            if block_stride != 1:
                block_downsample = nn.Sequential(
                    nn.Conv1d(
                        self.in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=block_stride,
                        bias=False
                    ),
                    nn.BatchNorm1d(out_channels)
                )
            
            layers.append(
                block(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    stride=block_stride,
                    downsample=block_downsample,
                    dropout_rate=self.dropout_rate
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass
        
        Input: (batch_size, 1, 9000) - single-lead ECG
        Output: (batch_size, 4) - class logits
        
        Hannun et al.: "network takes as input raw ECG data"
        """
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, length) -> (batch, 1, length)
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Final classification layer
        x = self.fc(x)
        
        return x