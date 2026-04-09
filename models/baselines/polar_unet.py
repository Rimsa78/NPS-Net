"""
Polar-UNet baseline (B4).

Same polar input as NPS-Net but predicts polar masks DIRECTLY
(no boundary distributions, no by-construction nesting).

This model isolates whether gains come from the polar transform alone
vs the full NPS-Net formulation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CircularPadConv2d(nn.Module):
    """Conv2d with circular padding along θ (width) and zero-pad along ρ (height)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
        super().__init__()
        self.pad_rho = kernel_size // 2
        self.pad_theta = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.pad_theta, self.pad_theta, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, self.pad_rho, self.pad_rho), mode='constant', value=0)
        return self.conv(x)


class CircularDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            CircularPadConv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CircularPadConv2d(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PolarEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = CircularDoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.double_conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class PolarDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.double_conv = CircularDoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)


class PolarSamplingGrid(nn.Module):
    """Warp Cartesian image to polar coordinates."""

    def __init__(self, image_size, n_theta, n_rho):
        super().__init__()
        self.image_size = image_size
        self.n_theta = n_theta
        self.n_rho = n_rho

        H = W = image_size
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        R = min(H, W) / 2.0

        theta = torch.linspace(0, 2 * math.pi, n_theta + 1)[:n_theta]
        rho = torch.linspace(0, 1, n_rho)

        rho_2d = rho.unsqueeze(1)
        theta_2d = theta.unsqueeze(0)

        px = cx + R * rho_2d * torch.cos(theta_2d)
        py = cy + R * rho_2d * torch.sin(theta_2d)

        grid_x = 2.0 * px / (W - 1) - 1.0
        grid_y = 2.0 * py / (H - 1) - 1.0

        grid = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer('grid', grid.unsqueeze(0))
        self.register_buffer('_cx', torch.tensor(cx))
        self.register_buffer('_cy', torch.tensor(cy))
        self.register_buffer('_R', torch.tensor(R))

    def forward(self, image):
        B = image.shape[0]
        grid = self.grid.expand(B, -1, -1, -1)
        return F.grid_sample(image, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)


class PolarToCartesian(nn.Module):
    """Warp polar masks back to Cartesian space."""

    def __init__(self, image_size, n_theta, n_rho):
        super().__init__()
        H = W = image_size
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        R = min(H, W) / 2.0

        ys = torch.arange(H, dtype=torch.float32)
        xs = torch.arange(W, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

        rho = torch.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / R
        theta = torch.atan2(grid_y - cy, grid_x - cx) % (2 * math.pi)

        # Normalise to [-1, 1] for grid_sample on polar image
        # polar image dims: height=n_rho, width=n_theta
        grid_rho = 2.0 * rho - 1.0  # ρ ∈ [0,1] → [-1,1]
        grid_theta = 2.0 * theta / (2 * math.pi) - 1.0  # θ ∈ [0,2π) → [-1,1)

        # Mask pixels outside the inscribed circle
        inside = (rho <= 1.0).float()

        cart_grid = torch.stack([grid_theta, grid_rho], dim=-1)
        self.register_buffer('cart_grid', cart_grid.unsqueeze(0))
        self.register_buffer('inside', inside.unsqueeze(0).unsqueeze(0))

    def forward(self, polar_mask):
        """
        polar_mask: (B, C, N_ρ, N_θ) in logit or probability space
        Returns: (B, C, H, W)
        """
        B = polar_mask.shape[0]
        grid = self.cart_grid.expand(B, -1, -1, -1)
        cart = F.grid_sample(polar_mask, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)
        return cart * self.inside


class PolarUNet(nn.Module):
    """Polar-UNet: warp to polar → UNet with circular padding → warp back.

    Predicts polar cup/disc masks directly — NO boundary distributions,
    NO by-construction nesting.
    """

    def __init__(self, in_channels=3, out_channels=2,
                 image_size=512, n_theta=360, n_rho=256,
                 features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.image_size = image_size
        self.n_theta = n_theta
        self.n_rho = n_rho

        # Polar sampling
        self.polar_grid = PolarSamplingGrid(image_size, n_theta, n_rho)
        self.polar_to_cart = PolarToCartesian(image_size, n_theta, n_rho)

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(PolarEncoderBlock(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = CircularDoubleConv(features[-1], features[-1] * 2)

        # Decoder
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(PolarDecoderBlock(f * 2, f))

        # Output: 2 channels (cup, disc) as logits in polar space
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            logits: (B, 2, H, W) — channel 0 = cup, channel 1 = disc (Cartesian)
        """
        # Warp to polar
        polar = self.polar_grid(x)  # (B, 3, N_ρ, N_θ)

        # Encoder
        skips = []
        feat = polar
        for enc in self.encoders:
            skip, feat = enc(feat)
            skips.append(skip)

        # Bottleneck
        feat = self.bottleneck(feat)

        # Decoder
        skips_rev = skips[::-1]
        for idx, dec in enumerate(self.decoders):
            feat = dec(feat, skips_rev[idx])

        # Polar logits
        polar_logits = self.final_conv(feat)  # (B, 2, N_ρ, N_θ)

        # Warp back to Cartesian
        cart_logits = self.polar_to_cart(polar_logits)  # (B, 2, H, W)

        return cart_logits


if __name__ == "__main__":
    model = PolarUNet(in_channels=3, out_channels=2, image_size=512)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"PolarUNet | Input: {x.shape} → Output: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params:,}")
