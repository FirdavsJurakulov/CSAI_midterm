"""DINOv2 ViT-S/14 backbone + MLP head for brand classification."""
import timm
import torch
from torch import nn

BACKBONES = {"dinov2_vits14": "vit_small_patch14_dinov2.lvd142m"}


def make_backbone(name="dinov2_vits14", img_size=224, pretrained=True):
    return timm.create_model(BACKBONES[name], pretrained=pretrained, num_classes=0, img_size=img_size)


def embed(backbone, x):
    """CLS token concatenated with the mean of patch tokens (the DINOv2 linear-probe recipe): 2 x 384 = 768-d."""
    tokens = backbone.forward_features(x)
    cls = tokens[:, 0]
    patches = tokens[:, backbone.num_prefix_tokens:].mean(dim=1)
    return torch.cat([cls, patches], dim=1)


class BrandHead(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, f):
        return self.net(f)


class BrandNet(nn.Module):
    """Stage B: backbone with only its last `unfreeze_blocks` blocks trainable, plus the head.

    The frozen prefix runs under no_grad, so its activations are not stored: this is what fits a 4 GB GPU.
    """
    def __init__(self, backbone, head, unfreeze_blocks=4):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.n_frozen = len(backbone.blocks) - unfreeze_blocks
        for p in backbone.parameters():
            p.requires_grad = False
        for p in [*backbone.blocks[self.n_frozen:].parameters(), *backbone.norm.parameters()]:
            p.requires_grad = True

    def features(self, x):
        b = self.backbone
        with torch.no_grad():
            x = b.norm_pre(b.patch_drop(b._pos_embed(b.patch_embed(x))))
            for blk in b.blocks[:self.n_frozen]:
                x = blk(x)
        for blk in b.blocks[self.n_frozen:]:
            x = blk(x)
        x = b.norm(x)
        return torch.cat([x[:, 0], x[:, b.num_prefix_tokens:].mean(dim=1)], dim=1)

    def forward(self, x):
        return self.head(self.features(x))
