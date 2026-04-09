# dofe.py
"""
DoFE — Domain-oriented Feature Embedding for Generalizable Fundus Image Segmentation.

Reference:
    Wang et al., "DoFE: Domain-oriented Feature Embedding for Generalizable
    Fundus Image Segmentation on Unseen Datasets", IEEE TMI, 2020.

Architecture:
    1. ResNet-34 Encoder (from scratch — no pretrained weights)
    2. DoFE Module (Domain Knowledge Pool + attention-based feature mixing)
    3. Multi-scale Segmentation Decoder with skip connections
    4. Domain Classification Head (auxiliary task)

For single-source training, we use pseudo-domain labels computed via
K-means clustering on image statistics (LAB colour, contrast).

Output:
    forward(x) → (B, 2, H, W) logits — channel 0 = cup, channel 1 = disc
    forward_with_domain(x) → dict with seg_logits + domain_logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# ResNet-34 Building Blocks (shared with BEAL)
# ==============================================================================

class BasicBlock(nn.Module):
    """Standard ResNet basic block."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet34Encoder(nn.Module):
    """ResNet-34 encoder producing 4 stages of feature maps."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = self._make_stage(64,  64,  3, stride=1)
        self.stage2 = self._make_stage(64,  128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)

    def _make_stage(self, in_ch, out_ch, n_blocks, stride):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [BasicBlock(in_ch, out_ch, stride, downsample)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.stem(x)
        s1 = self.stage1(x0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return [s1, s2, s3, s4]


# ==============================================================================
# DoFE Module — Domain-oriented Feature Embedding
# ==============================================================================

class DomainKnowledgePool(nn.Module):
    """Learnable domain prototype bank.

    Stores K domain prototypes of dimension C. Uses attention to compute
    domain-aware feature re-weighting.
    """

    def __init__(self, feature_dim, n_domains=4):
        super().__init__()
        self.n_domains = n_domains
        self.feature_dim = feature_dim

        # Learnable domain prototypes: (K, C)
        self.prototypes = nn.Parameter(torch.randn(n_domains, feature_dim) * 0.02)

        # Attention projection
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) feature map

        Returns:
            attention_weights: (B, K) — similarity to each prototype
            domain_enhanced: (B, C, H, W) — re-weighted features
        """
        B, C, H, W = features.shape

        # Global average pool → (B, C)
        gap = features.mean(dim=[2, 3])

        # Compute attention between input and prototypes
        q = self.query_proj(gap)                           # (B, C)
        k = self.key_proj(self.prototypes)                 # (K, C)

        # Attention scores: (B, K)
        attn = torch.matmul(q, k.t()) / (C ** 0.5)
        attn_weights = F.softmax(attn, dim=1)

        # Weighted prototype: (B, C)
        weighted_proto = torch.matmul(attn_weights, self.prototypes)  # (B, C)

        # Channel re-weighting (SE-Net style)
        scale = torch.sigmoid(weighted_proto).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        domain_enhanced = features * scale

        return attn_weights, domain_enhanced


class DoFEModule(nn.Module):
    """Full Domain-oriented Feature Embedding module.

    Applies domain knowledge pool at the deepest feature level and
    uses residual connection to preserve information.
    """

    def __init__(self, feature_dim, n_domains=4):
        super().__init__()
        self.dkp = DomainKnowledgePool(feature_dim, n_domains)

        # Feature refinement after domain mixing
        self.refine = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) deepest encoder features

        Returns:
            attn_weights: (B, K) domain attention
            enhanced: (B, C, H, W) domain-enhanced features
        """
        attn_weights, enhanced = self.dkp(features)
        enhanced = self.refine(enhanced) + features  # residual
        return attn_weights, enhanced


# ==============================================================================
# Domain Classification Head
# ==============================================================================

class DomainClassifier(nn.Module):
    """Auxiliary domain classification head.

    Global average pool → FC → K-class prediction.
    Provides domain-aware gradients to the encoder.
    """

    def __init__(self, feature_dim, n_domains=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_domains),
        )

    def forward(self, features):
        """
        Args:
            features: (B, C, H, W) encoder features
        Returns:
            (B, K) domain logits
        """
        return self.head(features)


# ==============================================================================
# Segmentation Decoder
# ==============================================================================

class DecoderBlock(nn.Module):
    """Upsample + skip-cat + double conv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=2, stride=2)
        cat_ch = in_channels // 2 + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(cat_ch, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SegmentationDecoder(nn.Module):
    """Progressive upsampling decoder with skip connections."""

    def __init__(self, out_channels=2):
        super().__init__()
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64,  64)
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1),
        )

    def forward(self, features):
        s1, s2, s3, s4 = features
        x = self.dec4(s4, s3)
        x = self.dec3(x, s2)
        x = self.dec2(x, s1)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return self.final(x)


# ==============================================================================
# Full DoFE Model
# ==============================================================================

class DoFE(nn.Module):
    """DoFE: Domain-oriented Feature Embedding for Fundus Segmentation.

    Main forward: encoder → DoFE → decoder → (B, 2, H, W) seg logits.
    Auxiliary: domain classification head (used during training).
    """

    def __init__(self, in_channels=3, out_channels=2, n_domains=4):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels)
        self.dofe = DoFEModule(feature_dim=512, n_domains=n_domains)
        self.seg_decoder = SegmentationDecoder(out_channels)
        self.domain_head = DomainClassifier(feature_dim=512, n_domains=n_domains)

    def forward(self, x):
        """Standard forward: returns (B, 2, H, W) segmentation logits."""
        features = self.encoder(x)
        s1, s2, s3, s4 = features

        # Apply DoFE module on deepest features
        _, s4_enhanced = self.dofe(s4)

        # Decode with enhanced features
        return self.seg_decoder([s1, s2, s3, s4_enhanced])

    def forward_with_domain(self, x):
        """Training forward: returns seg logits + domain predictions.

        Returns:
            dict with:
                'seg_logits': (B, 2, H, W)
                'domain_logits': (B, K)
                'domain_attn': (B, K)
        """
        features = self.encoder(x)
        s1, s2, s3, s4 = features

        # DoFE module
        domain_attn, s4_enhanced = self.dofe(s4)

        # Segmentation
        seg_logits = self.seg_decoder([s1, s2, s3, s4_enhanced])

        # Domain classification (from original s4, not enhanced)
        domain_logits = self.domain_head(s4)

        return {
            'seg_logits': seg_logits,
            'domain_logits': domain_logits,
            'domain_attn': domain_attn,
        }


if __name__ == "__main__":
    model = DoFE(in_channels=3, out_channels=2, n_domains=4)
    x = torch.randn(2, 3, 512, 512)

    # Standard forward
    out = model(x)
    print(f"DoFE | Input: {x.shape} → Seg Output: {out.shape}")

    # Training forward
    out_full = model.forward_with_domain(x)
    print(f"DoFE | Seg: {out_full['seg_logits'].shape}, "
          f"Domain: {out_full['domain_logits'].shape}, "
          f"Attn: {out_full['domain_attn'].shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params:,}")
