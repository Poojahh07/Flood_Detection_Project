
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# 1. ATTENTION GATE
# --------------------------------------------------

class AttentionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.gate(x)
        return x * attention_weights


# --------------------------------------------------
# 2. FLOOD DETECTION TRANSFORMER
# --------------------------------------------------

class FloodTransformer(nn.Module):
    def __init__(
        self,
        sar_channels=2,
        optical_channels=13,
        num_classes=2,
        image_size=128
    ):
        super().__init__()

        self.image_size = image_size

        # SAR feature extraction
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(sar_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Optical feature extraction
        self.optical_encoder = nn.Sequential(
            nn.Conv2d(
                optical_channels, 32, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Fuse SAR and optical features: 64 + 64 = 128
        self.attention = AttentionGate(channels=128)

        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=2
        )

        # Pixel-wise classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    # --------------------------------------------------
    # 3. FORWARD PASS
    # --------------------------------------------------

    def forward(self, sar, optical):

        if sar.ndim != 4 or optical.ndim != 4:
            raise ValueError(
                "SAR and optical inputs must have shape "
                "[batch, channels, height, width]."
            )

        if sar.shape[0] != optical.shape[0]:
            raise ValueError(
                "SAR and optical batch sizes must match."
            )

        if sar.shape[-2:] != optical.shape[-2:]:
            raise ValueError(
                "SAR and optical images must have matching "
                "spatial dimensions."
            )

        # Extract features from both images
        sar_features = self.sar_encoder(sar)
        optical_features = self.optical_encoder(optical)

        # Combine the two modalities
        features = torch.cat(
            [sar_features, optical_features],
            dim=1
        )

        # Apply attention
        features = self.attention(features)

        batch, channels, height, width = features.shape

        # Convert spatial features into Transformer tokens
        tokens = features.flatten(2).transpose(1, 2)

        # Learn relationships between image locations
        tokens = self.transformer(tokens)

        # Convert tokens back into a feature map
        features = tokens.transpose(1, 2).reshape(
            batch, channels, height, width
        )

        # Predict two class scores for every pixel
        logits = self.classifier(features)

        # Restore the original input resolution
        logits = F.interpolate(
            logits,
            size=sar.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Shape: [batch, 2, height, width]
        return logits


# --------------------------------------------------
# 4. TEST THE MODEL
# --------------------------------------------------

if __name__ == "__main__":

    model = FloodTransformer()

    sar = torch.randn(1, 2, 128, 128)
    optical = torch.randn(1, 13, 128, 128)

    model.eval()

    with torch.no_grad():
        output = model(sar, optical)

    print("Model created successfully!")
    print("SAR input shape:", sar.shape)
    print("Optical input shape:", optical.shape)
    print("Output shape:", output.shape)
    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )

    assert output.shape == (1, 2, 128, 128)

    print("Model shape test passed!")
