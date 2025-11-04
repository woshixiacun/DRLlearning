import torch
import torch.nn as nn

class MyBlock(nn.Module):
    def __init__(self, c1, c2, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        c2 = c2 or c1  # 如果没有指定输出通道，则默认与输入通道相同
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.embed_dim = c1

        # 延迟初始化的 LayerNorm 与 MLP
        self.norm1 = None
        self.attn = None
        self.norm2 = None
        self.mlp = None

        # 1x1 conv 调整输出通道数（方便 YOLO 接上去）
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def _build(self, c):
        """根据实际输入通道 C 初始化注意力和 MLP层"""
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(embed_dim=c, num_heads=self.num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        hidden_dim = int(c * self.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(c, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, c)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        if self.attn is None:
            # 第一次 forward 时根据输入通道构建模块
            self._build(C)
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
