import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

from vmunet.vmunet import VMUNet


# Multi-Scale Convolution Module
class MultiScaleConv(nn.Module):
    def  __init__(self, in_channels, out_channels):
        """
        Multi-scale convolution with parallel branches using different kernel sizes.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after fusion.
        """
        super(MultiScaleConv, self).__init__()
        # Branch 1: Frequency-focused kernel (3x1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.SiLU()
        )
        # Branch 2: Time-focused kernel (1x3)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.SiLU()
        )
        # Branch 3: Joint frequency-time kernel (3x3)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.SiLU()
        )
        # Fusion layer to reduce concatenated channels back to out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(96, out_channels, kernel_size=(1, 1)),
            nn.SiLU()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        concat = torch.cat([b1, b2, b3], dim=1)  # Concatenate along channel dimension
        out = self.fusion(concat)
        return out


# Self-Attention Module along Time Axis
class TimeSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Multi-head self-attention applied along the time axis.

        Args:
            embed_dim (int): Feature dimension (number of channels).
            num_heads (int): Number of attention heads.
        """
        super(TimeSelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # Input shape: (N, C, H, W)
        N, C, H, W = x.shape
        # Permute to (W, N, H, C) for attention over time
        x = x.permute(3, 0, 2, 1)  # Time becomes sequence length
        # Reshape to (W, N*H, C) for MultiheadAttention
        x = x.reshape(W, N * H, C)
        # Apply self-attention
        attn_output, _ = self.attn(x, x, x)
        # Reshape back to (W, N, H, C)
        attn_output = attn_output.reshape(W, N, H, C)
        # Permute back to (N, C, H, W)
        attn_output = attn_output.permute(1, 3, 2, 0)
        return attn_output


# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, channels, num_heads):
        """
        Encoder block combining multi-scale convolution, self-attention, and downsampling.

        Args:
            channels (int): Number of channels (fixed throughout the block).
            num_heads (int): Number of attention heads.
        """
        super(EncoderBlock, self).__init__()
        self.multi_scale = MultiScaleConv(channels, channels)
        self.ln1 = nn.LayerNorm(channels)  # Normalize over channel dimension
        self.attn = TimeSelfAttention(channels, num_heads)
        self.ln2 = nn.LayerNorm(channels)
        # Downsample time dimension only
        self.downsample = nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

    def forward(self, x):
        # Residual connection around multi-scale convolution
        identity = x
        x = self.multi_scale(x)
        x = x + identity
        # Layer normalization (permute to move channels to last dim)
        x = self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Residual connection around self-attention
        identity = x
        x = self.attn(x)
        x = x + identity
        x = self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Downsample time dimension
        x = self.downsample(x)
        return x


# Full Spectrogram Encoder
class SpectrogramEncoder(nn.Module):
    def __init__(self, in_dim =20,out_dim=64, num_heads=8):
        """Spectrogram encoder mapping (N, 20, 64, 249) to (N, 30, 64, 64)."""
        super(SpectrogramEncoder, self).__init__()
        # Initial convolution to expand channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.SiLU()
        )
        # Two encoder blocks
        self.block1 = EncoderBlock(out_dim, num_heads=num_heads)
        self.block2 = EncoderBlock(out_dim, num_heads=num_heads)

    def forward(self, x):
        # Input: (N, 20, 64, 249)
        x = self.initial_conv(x)  # (N, 64, 64, 249)
        x = self.block1(x)  # (N, 64, 64, 125)
        x = self.block2(x)  # (N, 64, 64, 63)
        # Pad time dimension from 63 to 64
        x = F.pad(x, (0, 64-x.shape[-1], 0, 64-x.shape[-2]), mode='constant', value=0)  # (N, 64, 64, 64)
        return x



class fMRIDecoder(nn.Module):
    def __init__(self, in_dim: int=256, out_dim: int=30):
        """
        Args:
            out_dim (int): fMRI output channels
        """
        super().__init__()
        self.model = VMUNet(input_channels=in_dim,num_classes=out_dim)

    def forward(self, x):
        # Input: [256, W, H]
        x = self.model(x)
        return x

class Spectrogram2fMRI(nn.Module):
    def __init__(self, in_dim:int=20, out_dim:int=30, hidden_dim:int=256):
        super().__init__()
        self.encoder = SpectrogramEncoder(in_dim=in_dim,out_dim=hidden_dim)
        self.decoder = fMRIDecoder(in_dim=hidden_dim,out_dim=out_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Example Usage
if __name__ == "__main__":
    # Initialize model
    model = Spectrogram2fMRI(out_dim=32).cuda()
    # Create a sample input tensor
    input_tensor = torch.randn(1, 20, 43, 249).cuda()
    # Forward pass
    output_tensor = model(input_tensor)
    print(f"Output shape: {output_tensor.shape}")  # Expected: torch.Size([1, 30, 64, 64])

    torchinfo.summary(model, input_size=(1, 20, 43, 249))
    #
    # torch.Size([32, 20, 64, 249])
    # torch.Size([32, 30, 64, 64])
# EEG shape: (144, 20, 43, 249)
# fMRI shape: (144, 32, 64, 64)