import torch
import torch.nn as nn
import torch.nn.functional as F

from models import create_unet
from vmunet.vmunet import VMUNet


# Multi-Scale Convolution Module
class MultiScaleConv(nn.Module):
    def  __init__(self, in_channels, out_channels, ablation):
        """
        Multi-scale convolution with parallel branches using different kernel sizes.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after fusion.
            ablation (dict): Ablation settings to enable/disable specific branches.
        """
        super(MultiScaleConv, self).__init__()
        fusion_in_channels = 0

        # Branch 1: Frequency-focused kernel (3x1)
        if ablation["use_multi_scale_freq"]:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=(3, 1), padding=(1, 0)),
                nn.SiLU()
            )
            fusion_in_channels += 32
        # Branch 2: Time-focused kernel (1x3)
        if ablation["use_multi_scale_time"]:
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=(1, 3), padding=(0, 1)),
                nn.SiLU()
            )
            fusion_in_channels += 32
        # Branch 3: Joint frequency-time kernel (3x3)
        if ablation["use_multi_scale_joint"]:
            self.branch3 = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=(1, 1)),
                nn.SiLU()
            )
            fusion_in_channels += 32
        # Fusion layer to reduce concatenated channels back to out_channels

        if fusion_in_channels == 0:
            raise ValueError("At least one branch must be enabled in ablation settings.")

        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, out_channels, kernel_size=(1, 1)),
            nn.SiLU()
        )

        self.ablation = ablation

    def forward(self, x):
        # Apply each branch and concatenate results
        branches_res = []
        if self.ablation["use_multi_scale_freq"]:
            branches_res.append(self.branch1(x))
        if self.ablation["use_multi_scale_time"]:
            branches_res.append(self.branch2(x))
        if self.ablation["use_multi_scale_joint"]:
            branches_res.append(self.branch3(x))
        concat = torch.cat(branches_res, dim=1)  # Concatenate along channel dimension
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
    def __init__(self, channels, num_heads, ablation):
        """
        Encoder block combining multi-scale convolution, self-attention, and downsampling.

        Args:
            channels (int): Number of channels (fixed throughout the block).
            num_heads (int): Number of attention heads.
            ablation (dict): Ablation settings to enable/disable components.
        """
        super(EncoderBlock, self).__init__()
        self.multi_scale = MultiScaleConv(channels, channels, ablation)
        self.ln1 = nn.LayerNorm(channels)  # Normalize over channel dimension

        if ablation["use_self_attention"]:
            self.attn = TimeSelfAttention(channels, num_heads)
            self.ln2 = nn.LayerNorm(channels)


        # Downsample time dimension only
        self.downsample = nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        self.ablation = ablation

    def forward(self, x):
        # Residual connection around multi-scale convolution
        identity = x
        x = self.multi_scale(x)
        x = x + identity
        # Layer normalization (permute to move channels to last dim)
        x = self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Residual connection around self-attention
        if self.ablation["use_self_attention"]:
            identity = x
            x = self.attn(x)
            x = x + identity
            x = self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Downsample time dimension
        x = self.downsample(x)
        return x


# Full Spectrogram Encoder
class SpectrogramEncoder(nn.Module):
    def __init__(self, in_dim =20,out_dim=64, num_heads=8, ablation: dict = {
        "use_multi_scale_time": True,
        "use_multi_scale_freq": True,
        "use_multi_scale_joint": True,
        "use_self_attention": True
    }):
        """Spectrogram encoder mapping (N, 20, 64, 249) to (N, 30, 64, 64)."""
        super(SpectrogramEncoder, self).__init__()
        # Initial convolution to expand channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), padding=(1, 1)),
            nn.SiLU()
        )
        # Two encoder blocks
        self.block1 = EncoderBlock(out_dim, num_heads=num_heads, ablation = ablation)
        self.block2 = EncoderBlock(out_dim, num_heads=num_heads, ablation = ablation)

    def forward(self, x):
        # Input: (N, 20, 64, 249)
        x = self.initial_conv(x)  # (N, 64, 64, 249)
        x = self.block1(x)  # (N, 64, 64, 125)
        x = self.block2(x)  # (N, 64, 64, 63)
        # Pad time dimension from 63 to 64
        x = F.pad(x, (0, 64-x.shape[-1], 0, 64-x.shape[-2]), mode='constant', value=0)  # (N, 64, 64, 64)
        return x



class fMRIDecoder(nn.Module):
    def __init__(self, in_dim: int=256, out_dim: int=30, ablation:bool=False):
        """
        Args:
            out_dim (int): fMRI output channels
        """
        super().__init__()
        if ablation:
            self.model = create_unet(in_channels=in_dim, out_channels=out_dim)
        else:
            self.model = VMUNet(input_channels=in_dim,num_classes=out_dim)

        self.ablation = ablation

    def forward(self, x):
        # Input: [256, W, H]
        if self.ablation:
            bs = x.shape[0]
            timesteps = torch.zeros((bs,), device=x.device).long()
            x = self.model(sample = x, timestep = timesteps).sample
        else:
            x = self.model(x)
        return x

class Spectrogram2fMRI(nn.Module):
    def __init__(self, in_dim:int=20, out_dim:int=30, hidden_dim:int=256, ablation_decoder:bool=False, ablation_encoder:dict = {
        "use_multi_scale_time": True,
        "use_multi_scale_freq": True,
        "use_multi_scale_joint": True,
        "use_self_attention": True
    }):
        super().__init__()
        self.encoder = SpectrogramEncoder(in_dim=in_dim,out_dim=hidden_dim, ablation=ablation_encoder)
        self.decoder = fMRIDecoder(in_dim=hidden_dim,out_dim=out_dim, ablation=ablation_decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Example Usage
if __name__ == "__main__":
    import torchinfo

    # Initialize model
    model = Spectrogram2fMRI(out_dim=30, ablation_encoder = {
        "use_multi_scale_time": True,
        "use_multi_scale_freq": True,
        "use_multi_scale_joint": False,
        "use_self_attention": False
    }).cuda()
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