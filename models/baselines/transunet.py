import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from einops import rearrange

# ─────────────────────────────────────────────
# Patch Embedding
# ─────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """
    Splits the CNN feature map into fixed-size patches
    and linearly projects each patch to a d_model-dim token.
    """
    def __init__(self, in_channels, patch_size, d_model):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, d_model,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                          # (B, d_model, H/p, W/p)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # (B, N, d_model)
        return x, H, W


# ─────────────────────────────────────────────
# Multi-Head Self-Attention
# ─────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_drop_p = attn_drop

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)         # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Use memory-efficient / Flash Attention kernel (O(N) memory instead of O(N²))
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# ─────────────────────────────────────────────
# MLP Block (inside Transformer)
# ─────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio=4.0, drop=0.0):
        super(MLP, self).__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, d_model),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# Transformer Encoder Block
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0,
                 attn_drop=0.0, mlp_drop=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, mlp_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# Transformer Encoder (stack of blocks)
# ─────────────────────────────────────────────
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers,
                 mlp_ratio=4.0, attn_drop=0.0, mlp_drop=0.0,
                 use_grad_checkpoint=False):
        super(TransformerEncoder, self).__init__()
        self.use_grad_checkpoint = use_grad_checkpoint
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio, attn_drop, mlp_drop)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            if self.use_grad_checkpoint and self.training:
                x = grad_checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm(x)


# ─────────────────────────────────────────────
# CNN Encoder (Hybrid backbone)
# ─────────────────────────────────────────────
class CNNEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNEncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


# ─────────────────────────────────────────────
# CNN Decoder Block
# ─────────────────────────────────────────────
class CNNDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNDecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
# Full TransUNet
# ─────────────────────────────────────────────
class TransUNet(nn.Module):
    """
    TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
    Chen et al., 2021  |  https://arxiv.org/abs/2102.04306

    Architecture:
        1. Hybrid CNN Encoder  →  extracts multi-scale feature maps + skip connections
        2. Patch Embedding     →  tokenises the deepest CNN feature map
        3. Transformer Encoder →  captures global context via self-attention
        4. Reshape + Project   →  restores spatial layout from transformer tokens
        5. CNN Decoder         →  upsamples with skip connections (U-Net style)
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        img_size=256,
        cnn_features=[64, 128, 256],   # CNN encoder stage output channels
        d_model=512,                   # Transformer hidden dim (= deepest CNN output)
        patch_size=1,                  # Patch size on the CNN feature map (usually 1)
        num_heads=8,
        num_layers=6,                  # reduced from 12 for memory efficiency
        mlp_ratio=4.0,
        attn_drop=0.1,
        mlp_drop=0.1,
        use_grad_checkpoint=True,      # trade compute for memory
    ):
        super(TransUNet, self).__init__()

        self.cnn_features = cnn_features
        self.d_model = d_model

        # ── 1. CNN Encoder ──────────────────────────────────────────────────
        self.cnn_encoders = nn.ModuleList()
        ch = in_channels
        for feat in cnn_features:
            self.cnn_encoders.append(CNNEncoderBlock(ch, feat))
            ch = feat

        # Project last CNN feature map to d_model channels before tokenisation
        self.channel_proj = nn.Sequential(
            nn.Conv2d(cnn_features[-1], d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

        # ── 2. Patch Embedding ───────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(d_model, patch_size, d_model)

        # Positional embedding (learnable); size determined at runtime
        # We register it as a parameter after computing sequence length
        # Using a large enough positional table: (img_size / 2^num_cnn_stages)^2
        num_cnn_stages = len(cnn_features)
        feat_map_size = img_size // (2 ** num_cnn_stages)
        num_patches = (feat_map_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(mlp_drop)

        # ── 3. Transformer Encoder ───────────────────────────────────────────
        self.transformer = TransformerEncoder(
            d_model, num_heads, num_layers, mlp_ratio, attn_drop, mlp_drop,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        # ── 4. Reshape + Project back to spatial feature map ─────────────────
        self.reshape_proj = nn.Sequential(
            nn.Conv2d(d_model, cnn_features[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(cnn_features[-1]),
            nn.ReLU(inplace=True)
        )

        # First upsample: project channels from cnn_features[-1] to cnn_features[-1] * 2
        # This prepares the feature map for the decoder which expects that many channels
        self.trans_up = nn.Sequential(
            nn.Conv2d(cnn_features[-1], cnn_features[-1] * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(cnn_features[-1] * 2),
            nn.ReLU(inplace=True)
        )

        # ── 5. CNN Decoder ───────────────────────────────────────────────────
        # Decoder channels: reversed CNN features
        # Input to first decoder block = transformer output + skip from last CNN encoder
        # Since transformer output goes through reshape_proj (d_model → cnn_features[-1]),
        # the first decoder input is cnn_features[-1] * 2 (transformer + skip)
        decoder_in_channels = [cnn_features[-1] * 2] + \
                               [cnn_features[i] * 2 for i in range(len(cnn_features) - 2, -1, -1)]
        decoder_out_channels = list(reversed(cnn_features))

        self.cnn_decoders = nn.ModuleList()
        for i in range(len(cnn_features)):
            self.cnn_decoders.append(
                CNNDecoderBlock(decoder_in_channels[i], decoder_out_channels[i])
            )

        # Final output - no extra upsampling needed since we have 3 decoders for 3 encoders
        # The decoder already outputs at img_size resolution
        self.final_up = nn.Identity()
        self.final_conv = nn.Sequential(
            nn.Conv2d(cnn_features[0], cnn_features[0] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_features[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_features[0] // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # ── CNN Encoder ──────────────────────────────────────────────────────
        skip_connections = []
        for encoder in self.cnn_encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)          # save skip connections

        # ── Channel projection ───────────────────────────────────────────────
        x = self.channel_proj(x)                   # (B, d_model, H', W')

        # ── Patch Embedding + Positional Encoding ────────────────────────────
        tokens, H_feat, W_feat = self.patch_embed(x)   # (B, N, d_model)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        # ── Transformer Encoder ──────────────────────────────────────────────
        tokens = self.transformer(tokens)               # (B, N, d_model)

        # ── Reshape tokens back to spatial feature map ───────────────────────
        x = rearrange(tokens, 'b (h w) c -> b c h w', h=H_feat, w=W_feat)
        x = self.reshape_proj(x)                        # (B, cnn_features[-1], H', W')

        # First upsample (bottleneck → scale of last CNN encoder skip)
        x = self.trans_up(x)

        # ── CNN Decoder ──────────────────────────────────────────────────────
        skip_connections = skip_connections[::-1]       # reverse: deepest first
        for idx, decoder in enumerate(self.cnn_decoders):
            x = decoder(x, skip_connections[idx])

        # ── Final output ─────────────────────────────────────────────────────
        x = self.final_up(x)
        return self.final_conv(x)


if __name__ == "__main__":
    model = TransUNet(
        in_channels=3,
        out_channels=1,
        img_size=512,
        cnn_features=[64, 128, 256],
        d_model=512,
        patch_size=1,
        num_heads=8,
        num_layers=6,
    )
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"TransUNet | Input: {x.shape} → Output: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {total_params:,}")
