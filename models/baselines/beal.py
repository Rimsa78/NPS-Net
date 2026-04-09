# beal.py
"""
BEAL — Boundary and Entropy-driven Adversarial Learning for Fundus Image Segmentation.

Reference:
    Wang et al., "Boundary and Entropy-driven Adversarial Learning for Fundus
    Image Segmentation on Optic Disc and Cup", IEEE TMI, 2019.

Architecture:
    1. ResNet-34 Encoder (from scratch — no pretrained weights for fair comparison)
    2. Multi-scale Segmentation Decoder with skip connections
    3. Boundary Detection Branch (cup + disc edge supervision)
    4. Adversarial Discriminator (separate module, separate optimiser)

Output:
    forward(x) → (B, 2, H, W) logits — channel 0 = cup, channel 1 = disc
    forward_with_boundary(x) → dict with seg_logits + boundary_maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# ResNet-34 Building Blocks
# ==============================================================================

class BasicBlock(nn.Module):
    """Standard ResNet basic block: two 3×3 convs with residual shortcut."""
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
    """ResNet-34 encoder producing 4 stages of feature maps.

    Stage outputs (for input 512×512):
        stage1: (B, 64,  128, 128)
        stage2: (B, 128,  64,  64)
        stage3: (B, 256,  32,  32)
        stage4: (B, 512,  16,  16)
    """

    def __init__(self, in_channels=3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Stages (ResNet-34: [3, 4, 6, 3] blocks)
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
        x0 = self.stem(x)           # (B, 64, H/4, W/4)
        s1 = self.stage1(x0)        # (B, 64, H/4, W/4)
        s2 = self.stage2(s1)        # (B, 128, H/8, W/8)
        s3 = self.stage3(s2)        # (B, 256, H/16, W/16)
        s4 = self.stage4(s3)        # (B, 512, H/32, W/32)
        return [s1, s2, s3, s4]


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
        # Handle size mismatch
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SegmentationDecoder(nn.Module):
    """Progressive upsampling decoder with skip connections from encoder.

    Takes 4-stage features [s1, s2, s3, s4] and produces (B, out_ch, H/4, W/4)
    then upsamples to original resolution.
    """

    def __init__(self, out_channels=2):
        super().__init__()
        # Decoder blocks: s4→s3→s2→s1
        self.dec4 = DecoderBlock(512, 256, 256)     # 512→256, cat 256 → 256 out
        self.dec3 = DecoderBlock(256, 128, 128)     # 256→128, cat 128 → 128 out
        self.dec2 = DecoderBlock(128, 64,  64)      # 128→64,  cat 64  → 64 out

        # Final upsample from H/4 → H (factor 4)
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1),
        )

    def forward(self, features):
        s1, s2, s3, s4 = features

        x = self.dec4(s4, s3)    # (B, 256, H/16, W/16)
        x = self.dec3(x, s2)     # (B, 128, H/8, W/8)
        x = self.dec2(x, s1)     # (B, 64, H/4, W/4)

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return self.final(x)


# ==============================================================================
# Boundary Detection Branch
# ==============================================================================

class BoundaryBranch(nn.Module):
    """Lightweight boundary prediction branch.

    Takes encoder features and produces (B, 2, H, W) boundary probability maps
    for cup and disc boundaries respectively.
    """

    def __init__(self, out_channels=2):
        super().__init__()
        # Fuse multi-scale features
        self.reduce_s1 = nn.Conv2d(64,  32, 1)
        self.reduce_s2 = nn.Conv2d(128, 32, 1)
        self.reduce_s3 = nn.Conv2d(256, 32, 1)
        self.reduce_s4 = nn.Conv2d(512, 32, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, features, target_size):
        """
        Args:
            features: [s1, s2, s3, s4] from encoder
            target_size: (H, W) of the original image
        Returns:
            (B, 2, H, W) boundary logits
        """
        s1, s2, s3, s4 = features
        H, W = target_size

        f1 = F.interpolate(self.reduce_s1(s1), (H, W), mode='bilinear', align_corners=True)
        f2 = F.interpolate(self.reduce_s2(s2), (H, W), mode='bilinear', align_corners=True)
        f3 = F.interpolate(self.reduce_s3(s3), (H, W), mode='bilinear', align_corners=True)
        f4 = F.interpolate(self.reduce_s4(s4), (H, W), mode='bilinear', align_corners=True)

        fused = torch.cat([f1, f2, f3, f4], dim=1)  # (B, 128, H, W)
        return self.fuse(fused)


# ==============================================================================
# Adversarial Discriminator
# ==============================================================================

class Discriminator(nn.Module):
    """PatchGAN-style discriminator for BEAL adversarial training.

    Input: (B, 2, H, W) — either predicted or GT masks (cup + disc channels).
    Output: (B, 1) real/fake score.
    """

    def __init__(self, in_channels=2):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 2, 512, 512) → (B, 32, 256, 256)
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # → (B, 64, 128, 128)
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # → (B, 128, 64, 64)
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # → (B, 256, 32, 32)
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # → (B, 512, 16, 16)
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Global average pool + classifier
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# Full BEAL Model
# ==============================================================================

class BEAL(nn.Module):
    """BEAL: Boundary and Entropy-driven Adversarial Learning.

    Main forward path: encoder → decoder → (B, 2, H, W) seg logits.
    Auxiliary: boundary branch + discriminator (used only during training).
    """

    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels)
        self.seg_decoder = SegmentationDecoder(out_channels)
        self.boundary_branch = BoundaryBranch(out_channels)

    def forward(self, x):
        """Standard forward: returns (B, 2, H, W) segmentation logits."""
        features = self.encoder(x)
        return self.seg_decoder(features)

    def forward_with_boundary(self, x):
        """Training forward: returns seg logits + boundary maps.

        Returns:
            dict with:
                'seg_logits': (B, 2, H, W)
                'boundary_logits': (B, 2, H, W)
        """
        H, W = x.shape[2], x.shape[3]
        features = self.encoder(x)
        seg_logits = self.seg_decoder(features)
        bnd_logits = self.boundary_branch(features, (H, W))
        return {
            'seg_logits': seg_logits,
            'boundary_logits': bnd_logits,
        }


if __name__ == "__main__":
    model = BEAL(in_channels=3, out_channels=2)
    disc = Discriminator(in_channels=2)
    x = torch.randn(2, 3, 512, 512)

    # Standard forward
    out = model(x)
    print(f"BEAL | Input: {x.shape} → Seg Output: {out.shape}")

    # Training forward
    out_full = model.forward_with_boundary(x)
    print(f"BEAL | Seg: {out_full['seg_logits'].shape}, "
          f"Bnd: {out_full['boundary_logits'].shape}")

    # Discriminator
    fake_masks = torch.sigmoid(out).detach()
    d_out = disc(fake_masks)
    print(f"Discriminator | Input: {fake_masks.shape} → Score: {d_out.shape}")

    total_gen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_disc = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Generator Params: {total_gen:,}")
    print(f"Discriminator Params: {total_disc:,}")
