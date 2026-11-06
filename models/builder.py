"""
Model builder for INP-Former-MG.

Turns a flat set of configuration options (usually parsed from CLI / a config
file) into a ready-to-train ``INP_Former`` model with per-granularity
prototype sets, extractors, bottlenecks and decoders.

The original INP-Former (single granularity) is recovered by using a single
layer group: ``layer_groups = [[0, 1, 2, 3, 4, 5, 6, 7]]``.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from . import vit_encoder
from .uad import INP_Former
from .vision_transformer import Mlp, Aggregation_Block, Prototype_Block


# Default intermediate layers to extract, keyed by encoder size.
TARGET_LAYERS = {
    "small": [2, 3, 4, 5, 6, 7, 8, 9],
    "base": [2, 3, 4, 5, 6, 7, 8, 9],
    "large": [4, 6, 8, 10, 12, 14, 16, 18],
}

EMBED_DIMS = {"small": 384, "base": 768, "large": 1024}
NUM_HEADS = {"small": 6, "base": 12, "large": 16}


def _encoder_size(name: str) -> str:
    for size in ("small", "base", "large"):
        if size in name:
            return size
    raise ValueError(f"Unknown encoder size in '{name}' (expected small/base/large)")


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x != ""]


def _parse_layer_groups(text: str) -> list[list[int]]:
    """Parse ``"0,1,2;3,4,5;6,7"`` into ``[[0,1,2],[3,4,5],[6,7]]``."""
    return [_parse_int_list(g) for g in text.replace(" ", "").split(";") if g != ""]


def build_model(
    encoder_name: str,
    layer_groups: str | list[list[int]],
    inp_nums: str | list[int],
    decoder_layers: str | list[int] | None = None,
    remove_class_token: bool = True,
    coherence_mode: str = "soft",
    coherence_temperature: float = 0.1,
    diversity_weight: float = 0.0,
    adaptive_prototypes: bool = False,
    gate_temperature: float = 1.0,
    gate_sparsity_weight: float = 0.0,
    num_agg_blocks: int = 1,
):
    """Build the multi-granularity INP-Former.

    Parameters
    ----------
    encoder_name : str
        e.g. ``"dinov2reg_vit_base_14"``.
    layer_groups : str | list[list[int]]
        Partition of ``target_layers`` (indices 0..7) into granularity groups.
        ``"0,1,2,3;4,5,6,7"`` -> 2 groups (shallow / deep).  A single group
        ``"0,1,2,3,4,5,6,7"`` recovers the original INP-Former.
    inp_nums : str | list[int]
        Number of INPs per group, e.g. ``"6,6"`` for two groups of 6.
    decoder_layers : str | list[int] | None
        Number of decoder blocks per group.  Defaults to the number of layers
        in the corresponding group (keeps total decoder depth == 8, matching
        the original).
    """
    encoder = vit_encoder.load(encoder_name)
    size = _encoder_size(encoder_name)
    embed_dim = EMBED_DIMS[size]
    num_heads = NUM_HEADS[size]
    target_layers = TARGET_LAYERS[size]

    if isinstance(layer_groups, str):
        layer_groups = _parse_layer_groups(layer_groups)
    if isinstance(inp_nums, str):
        inp_nums = _parse_int_list(inp_nums)
    if isinstance(decoder_layers, str):
        decoder_layers = _parse_int_list(decoder_layers)

    G = len(layer_groups)
    assert len(inp_nums) == G, \
        f"need one INP count per group: {len(inp_nums)} != {G}"

    if decoder_layers is None:
        decoder_layers = [len(g) for g in layer_groups]
    assert len(decoder_layers) == G, \
        f"need one decoder depth per group: {len(decoder_layers)} != {G}"

    groups = nn.ModuleList()
    prototype_tokens = nn.ParameterList()

    for g in range(G):
        inp_num = inp_nums[g]
        n_dec = decoder_layers[g]

        sub = nn.ModuleDict()
        # INP extractor: cross-attention blocks that aggregate normal tokens.
        sub["aggregation"] = nn.ModuleList([
            Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(num_agg_blocks)
        ])
        # Feature bottleneck.
        sub["bottleneck"] = nn.ModuleList([
            Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)
        ])
        # INP-guided decoder.
        sub["decoder"] = nn.ModuleList([
            Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(n_dec)
        ])
        groups.append(sub)

        # Learnable INPs for this granularity group (trunc-normal init, as in
        # the original INP-Former).
        prototype_tokens.append(nn.Parameter(torch.randn(inp_num, embed_dim)))

    model = INP_Former(
        encoder=encoder,
        groups=groups,
        prototype_tokens=prototype_tokens,
        target_layers=target_layers,
        layer_groups=layer_groups,
        remove_class_token=remove_class_token,
        coherence_mode=coherence_mode,
        coherence_temperature=coherence_temperature,
        diversity_weight=diversity_weight,
        adaptive_prototypes=adaptive_prototypes,
        gate_temperature=gate_temperature,
        gate_sparsity_weight=gate_sparsity_weight,
    )
    return model
