"""
Smoke test for INP-Former-MG core logic (no dataset / no DINOv2 download).

Builds the model with a tiny *mock* encoder and checks that

  1. single-granularity (G=1) and multi-granularity (G=3) both run forward,
  2. the output tensors have the correct shapes,
  3. the coherence / diversity / adaptive-gate terms are scalars,
  4. gradients flow back into every trainable sub-module (per group).

Run from the repo root:  python smoke_test.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block


class MockBlock(nn.Module):
    """A learnable stand-in for one ViT encoder block."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.proj(self.norm(x))


class MockEncoder(nn.Module):
    """Mimics the DINOv2 encoder interface used by INP_Former."""

    def __init__(self, dim=64, depth=12, img_size=224, patch=14, register=4):
        super().__init__()
        self.num_register_tokens = register
        self.embed_dim = dim
        self.patch = patch
        n_patches = (img_size // patch) ** 2
        self.n_patches = n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, register, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + register + n_patches, dim))
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.blocks = nn.ModuleList([MockBlock(dim) for _ in range(depth)])

    def prepare_tokens(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)      # [B, N, C]
        cls = self.cls_token.expand(B, -1, -1)
        reg = self.register_tokens.expand(B, -1, -1)
        tokens = torch.cat([cls, reg, patches], dim=1)
        return tokens + self.pos_embed


def _build(dim, layer_groups, inp_nums, **kw):
    embed_dim, num_heads = dim, 4
    groups = nn.ModuleList()
    protos = nn.ParameterList()
    dec = kw.pop("decoder_layers", None)
    for g in range(len(layer_groups)):
        n_dec = len(layer_groups[g]) if dec is None else dec[g]
        sub = nn.ModuleDict({
            "aggregation": nn.ModuleList([Aggregation_Block(embed_dim, num_heads, qkv_bias=True)]),
            "bottleneck": nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim)]),
            "decoder": nn.ModuleList([Prototype_Block(embed_dim, num_heads, qkv_bias=True)
                                      for _ in range(n_dec)]),
        })
        groups.append(sub)
        protos.append(nn.Parameter(torch.randn(inp_nums[g], embed_dim)))
    enc = MockEncoder(dim=embed_dim)
    return INP_Former(encoder=enc, groups=groups, prototype_tokens=protos,
                      layer_groups=layer_groups, target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
                      remove_class_token=True, **kw)


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)

    print("== Test 1: single granularity (G=1, == original structure) ==")
    model = _build(64, [[0, 1, 2, 3, 4, 5, 6, 7]], [6], coherence_mode="hard")
    en, de, loss = model(x)
    assert len(en) == len(de) == 1, f"expected 1 group, got {len(en)}"
    assert en[0].shape == de[0].shape == (2, 64, 16, 16), (en[0].shape, de[0].shape)
    assert loss.ndim == 0, "coherence loss must be scalar"
    print(f"  en/de shape {tuple(en[0].shape)}, coherence={loss.item():.4f}  OK")

    print("== Test 2: multi granularity (G=3, shallow/middle/deep) ==")
    model = _build(64, [[0, 1, 2], [3, 4, 5], [6, 7]], [4, 4, 4],
                   coherence_mode="soft", diversity_weight=0.05,
                   adaptive_prototypes=True)
    en, de, loss = model(x)
    assert len(en) == len(de) == 3, f"expected 3 groups, got {len(en)}"
    for e, d in zip(en, de):
        assert e.shape == d.shape == (2, 64, 16, 16), (e.shape, d.shape)
    assert loss.ndim == 0
    print(f"  3 groups, coherence+div+gate={loss.item():.4f}  OK")

    print("== Test 3: gradient flow into every group ==")
    model.train()
    en, de, loss = model(x)
    recon = sum((1 - torch.cosine_similarity(e, d)).mean() for e, d in zip(en, de))
    total = recon + 0.2 * loss
    total.backward()
    for g, group in enumerate(model.groups):
        for name, module in group.items():
            for p in module.parameters():
                assert p.grad is not None, f"no grad: group {g} / {name}"
    assert model.prototype_tokens[0].grad is not None
    print("  gradients flow into all per-group sub-modules  OK")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
