import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def compute_fft_batch(images):
    # Convert RGB to grayscale: (B, C, H, W) -> (B, H, W)
    gray = 0.2989 * images[:, 0] + 0.5870 * images[:, 1] + 0.1140 * images[:, 2]
    gray = gray.unsqueeze(1)  

    # Compute FFT
    f = torch.fft.fft2(gray)
    f = torch.fft.fftshift(f, dim=(-2, -1))
    mag = torch.abs(f).clamp_min(1e-8)
    log_mag = torch.log(mag)

    # Normalize
    b, c, h, w = log_mag.shape
    log_mag_flat = log_mag.view(b, c, -1)
    mn = log_mag_flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
    mx = log_mag_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
    freq = (log_mag - mn) / (mx - mn + 1e-8)

    return freq  

# Frame-Level Model Definitions 

class MultiScaleFrequencyBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.low_freq = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mid_freq = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.high_freq = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 128, 3, stride=2, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        low = self.low_freq(x)
        mid = self.mid_freq(x)
        high = self.high_freq(x)

        h, w = mid.shape[2], mid.shape[3]
        low = F.interpolate(low, size=(h, w), mode='bilinear', align_corners=False)
        high = F.interpolate(high, size=(h, w), mode='bilinear', align_corners=False)

        multi_scale = torch.cat([low, mid, high], dim=1)
        fused = self.fusion(multi_scale).flatten(1)
        return self.fc(fused)

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.conv(x)
        return x * attention

class EnhancedDeepFakeDetector(nn.Module):
    def __init__(self, freeze_backbone=True, spatial_dim=512, freq_dim=256):
        super().__init__()
        efficientnet = models.efficientnet_b0(weights=None) 

        if freeze_backbone:
            for param in list(efficientnet.features.parameters())[:-20]:
                param.requires_grad = False

        self.spatial = efficientnet.features
        self.spatial_attention = SpatialAttention(1280)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_fc = nn.Sequential(
            nn.Linear(1280, spatial_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(spatial_dim, spatial_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.freq_branch = MultiScaleFrequencyBranch(freq_dim)
        self.cross_attention = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, spatial_dim + freq_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, spatial_input, freq_input):
        # Spatial pathway
        s = self.spatial(spatial_input)
        s = self.spatial_attention(s)
        s = self.spatial_pool(s).flatten(1)
        s = self.spatial_fc(s)

        # Frequency pathway
        f = self.freq_branch(freq_input)

        # Concatenate features
        combined = torch.cat([s, f], dim=1)

        # Cross-attention
        attention_weights = self.cross_attention(combined)
        combined = combined * attention_weights

        output = self.fusion(combined)
        return output

# Video-Level Model Definitions 

class TemporalBiLSTM(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=512, num_layers=2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.3
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        pooled = torch.mean(out, dim=1) 
        return self.fc(pooled)

class FullDeepFakeDetector(nn.Module):
    def __init__(self, base_weights_path=None, device='cpu'):
        super().__init__()
        
        # 1. Load the Frame-Level Model
        self.frame_model = EnhancedDeepFakeDetector(freeze_backbone=True).to(device)
        
        if base_weights_path:
            state_dict = torch.load(base_weights_path, map_location=device)
            self.frame_model.load_state_dict(state_dict, strict=False)
        
        self.frame_model.eval()
        
        # 2. Replace the frame model's final classifier with an Identity
        self.frame_model.fusion = nn.Identity()
        
        # 3. Create the Temporal (LSTM) head
        self.temporal = TemporalBiLSTM(feature_dim=768, hidden_dim=512)

    def forward(self, frames_tensor):
        b, t, c, h, w = frames_tensor.shape
        
        # 1. Reshape to process all frames as one big batch
        frames_flat = frames_tensor.view(b * t, c, h, w)
        
        # 2. Get frequency features for the whole batch
        freq_in = compute_fft_batch(frames_flat)
        
        # 3. Get 768-dim features from the frozen frame model
        with torch.no_grad(): 
            features_flat = self.frame_model(frames_flat, freq_in)
        
        # 4. Reshape features back into a sequence
        features_seq = features_flat.view(b, t, -1)
        
        # 5. Pass the sequence to the LSTM to get the final video-level prediction
        output = self.temporal(features_seq)
        
        return output