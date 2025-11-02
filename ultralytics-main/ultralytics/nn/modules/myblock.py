import torch
import torch.nn as nn

class MyBlock(nn.Module):
    def __init__(self, c1, c2, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(c1)
        self.attn = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c1)
        hidden_dim = int(c1 * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(c1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, c1)
        )
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        # Self-Attention
        x1 = self.norm1(x_flat)
        attn_out, _ = self.attn(x1, x1, x1)
        x_flat = x_flat + attn_out
        
        # FeedForward
        x2 = self.norm2(x_flat)
        x_flat = x_flat + self.mlp(x2)
        
        # reshape
        x_out = x_flat.transpose(1, 2).view(B, C, H, W)
        return self.proj(x_out)
