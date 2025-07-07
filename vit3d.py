import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels=1, patch_size=(4, 4, 4), embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        x = x.flatten(2).transpose(1, 2)  # -> [B, N_patches, embed_dim]
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x):
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT3D(nn.Module):
    def __init__(self, in_channels=1, input_shape=(64, 128, 128), patch_size=(4, 4, 4),
                 embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0, num_classes=2):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, patch_size, embed_dim)

        D, H, W = input_shape
        d, h, w = patch_size
        num_patches = (D // d) * (H // h) * (W // w)

        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

class MaskedPatchEmbed3D(nn.Module):
    """
    将 3D 图像划分为 patch，并同时 downsample 掩膜。
    支持自动对 patch 进行填充，以确保掩膜 patch 不越界。
    """
    def __init__(self, in_channels=1, patch_size=(8, 8, 8), embed_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x, mask=None):
        """
        x: [B, C, D, H, W]
        mask: [B, 1, D, H, W]
        """
        # 自动 padding 保证图像能被完整切 patch
        x = self._pad_if_needed(x)
        x_embed = self.proj(x)  # [B, C_embed, D', H', W']
        x_embed = x_embed.flatten(2).transpose(1, 2)  # [B, N, C]

        if mask is not None:
            mask = self._pad_if_needed(mask)
            # 将 voxel 级 mask 转换为 patch 级 mask
            avg_pool = nn.AvgPool3d(kernel_size=self.patch_size, stride=self.patch_size)
            mask_patches = avg_pool(mask.float())  # [B, 1, D', H', W']
            mask_patches = (mask_patches > 0.5).squeeze(1).flatten(1)  # [B, N]
            return x_embed, mask_patches  # x_embed: [B, N, C], mask_patches: [B, N]
        else:
            return x_embed, None

    def _pad_if_needed(self, x):
        B, C, D, H, W = x.shape
        pd = (0, (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2])
        ph = (0, (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1])
        pz = (0, (self.patch_size[0] - D % self.patch_size[0]) % self.patch_size[0])
        return F.pad(x, pd + ph + pz)  # F.pad 是从最后一维开始 pad


class MaskedViT3D(nn.Module):
    """
    一个支持 3D 掩膜区域建模的 Vision Transformer 模型。
    适用于医学图像分类等任务。
    """
    def __init__(self, 
                 in_channels=1, 
                 input_shape=(64, 64, 64), 
                 patch_size=(4, 4, 4), 
                 embed_dim=128,
                 depth=4, 
                 num_heads=4, 
                 num_classes=2):
        super().__init__()
        self.patch_embed = MaskedPatchEmbed3D(in_channels, patch_size, embed_dim)

        D, H, W = input_shape
        d, h, w = patch_size
        num_patches = math.ceil(D / d) * math.ceil(H / h) * math.ceil(W / w)

        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask=None):
        x_embed, mask_patch = self.patch_embed(x, mask)  # [B, N, C], [B, N]
        all_pos = self.pos_embed.pos_embed[:, 1:]  # [1, N, C]
        cls_token = self.pos_embed.cls_token  # [1, 1, C]
        cls_pos = self.pos_embed.pos_embed[:, :1]  # [1, 1, C]

        output_tokens = []
        for i in range(x.shape[0]):
            if mask_patch is not None:
                valid_tokens = x_embed[i][mask_patch[i]]  # [N_valid_i, C]
                valid_pos = all_pos[0, mask_patch[i]].unsqueeze(0)
            else:
                valid_tokens = x_embed[i]
                valid_pos = all_pos

            tokens = torch.cat([cls_token, valid_tokens.unsqueeze(0)], dim=1)
            pos_tokens = torch.cat([cls_pos, valid_pos], dim=1)
            tokens = tokens + pos_tokens
            tokens = self.blocks(tokens)
            tokens = self.norm(tokens)
            cls_final = tokens[:, 0]  # [1, C]
            output_tokens.append(cls_final)

        out = torch.cat(output_tokens, dim=0)  # [B, C]
        return self.head(out)


# if __name__ == "__main__":
#     model = MaskedViT3D()
#     x = torch.randn(2, 1, 64, 64, 64)
#     mask = torch.zeros_like(x)
#     mask[:, :, 20:50, 20:50, 20:50] = 1  # 关注局部区域
#     out = model(x, mask)
#     print(out.shape)  # -> [2, num_classes]