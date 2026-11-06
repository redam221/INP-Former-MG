"""
Multi-granularity INP-Former (INP-Former-MG).

An architecture-level extension of INP-Former (CVPR 2025, "Exploring Intrinsic
Normal Prototypes within a Single Image for Universal Anomaly Detection").

----------------------------------------------------------------------------
Motivation (what we change and why)
----------------------------------------------------------------------------
The original INP-Former extracts a *single* set of Intrinsic Normal Prototypes
(INPs) shared across *all* encoder depths.  Concretely, it

  1. takes ``target_layers = [2..9]`` (8 intermediate ViT layers),
  2. fuses them with a plain ``mean`` into one feature tensor,
  3. aggregates a *single* set of ``INP_num`` learnable prototypes from it via
     one cross-attention ``Aggregation_Block``,
  4. guides the *entire* decoder with that one prototype set.

This collapses shallow layers (local texture / edges) and deep layers (part /
semantic layout) into a single prototype space.  Shallow and deep anomalies
have very different statistics, so a single prototype set is a representational
bottleneck.

INP-Former-MG instead assigns an **independent prototype set, independent
aggregation, and independent decoder to each depth group** defined by
``layer_groups``.  The ``target_layers`` are partitioned into ``G`` granularity
groups (e.g. shallow / middle / deep), and each group ``g`` owns:

  - ``prototype_tokens[g]``    : its own learnable INPs,
  - ``groups[g].aggregation``  : its own INP extractor,
  - ``groups[g].bottleneck``   : its own feature bottleneck,
  - ``groups[g].decoder``      : its own INP-guided decoder.

Reconstruction is performed *per granularity*, and the per-group anomaly maps
are fused into a final map.  With ``G == 1`` the model reduces exactly to the
original INP-Former (single granularity), which makes the single-vs-multi
granularity ablation trivial.

Three auxiliary improvements are also provided (all optional / configurable):

  * **Soft-assignment INP coherence loss** (``coherence_mode='soft'``): the
    original ``gather_loss`` hard-assigns every normal token to its *nearest*
    prototype (``torch.min``), which blocks gradient flow to the other
    prototypes.  We replace it with a temperature-softened assignment so every
    prototype receives gradient.

  * **Prototype diversity regularizer** (``diversity_weight > 0``): a penalty
    on inter-prototype cosine similarity that prevents prototype collapse
    (several INPs degenerating to the same normal mode).

  * **Adaptive prototype gating** (``adaptive_prototypes=True``): a learned
    per-prototype logit + entropy regularization so the model can *switch off*
    redundant prototypes per granularity group, replacing the hand-tuned
    ``INP_num``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Prototype-related losses / regularizers
# ---------------------------------------------------------------------------
def soft_coherence_loss(
    x: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Soft-assignment INP coherence loss.

    ``x``          : [B, N, C] normal feature tokens.
    ``prototypes`` : [B, K, C] current (aggregated) prototype tokens.

    Every normal token is softly assigned to *all* prototypes via a
    temperature-scaled cosine softmax, and the weighted cosine distance is
    minimized.  Unlike the hard ``torch.min`` version, gradient flows to every
    prototype, which (a) helps the prototype set cover the normal manifold and
    (b) stabilises training.

    Returns a scalar loss.
    """
    xn = F.normalize(x, dim=-1)                    # [B, N, C]
    pn = F.normalize(prototypes, dim=-1)           # [B, K, C]
    sim = torch.einsum("bnc,bkc->bnk", xn, pn)     # [B, N, K] cosine sim
    dist = 1.0 - sim                               # [B, N, K]
    w = F.softmax(sim / max(temperature, 1e-6), dim=-1)   # soft assignment
    return (w * dist).sum(dim=-1).mean()           # scalar


def hard_coherence_loss(x: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Original INP-Former ``gather_loss`` (hard nearest-prototype assignment).

    ``x``          : [B, N, C] normal feature tokens.
    ``prototypes`` : [B, K, C] current (aggregated) prototype tokens.
    """
    xn = F.normalize(x, dim=-1)                    # [B, N, C]
    pn = F.normalize(prototypes, dim=-1)           # [B, K, C]
    sim = torch.einsum("bnc,bkc->bnk", xn, pn)     # [B, N, K] cosine sim
    dist = 1.0 - sim                               # [B, N, K]
    return dist.min(dim=-1).values.mean()          # scalar


def prototype_diversity_loss(prototypes: torch.Tensor) -> torch.Tensor:
    """Penalise inter-prototype cosine similarity to prevent collapse.

    ``prototypes`` : [B, K, C].  We penalise every positive off-diagonal cosine
    similarity, so the K prototypes are pushed to cover distinct normal modes.
    """
    B, K, _ = prototypes.shape
    if K <= 1:
        return prototypes.new_zeros(())
    pn = F.normalize(prototypes, dim=-1)           # [B, K, C]
    sim = torch.einsum("bkc,bmc->bkm", pn, pn)     # [B, K, K]
    eye = torch.eye(K, device=prototypes.device, dtype=sim.dtype)
    off = sim - eye                                # zero out the diagonal
    return off.clamp(min=0.0).sum() / (B * K * (K - 1))


def adaptive_prototype_gate(
    prototypes: torch.Tensor,
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft-gate a prototype set through a learned per-prototype logit.

    ``prototypes`` : [B, K, C] (aggregated) prototypes.
    ``logits``     : [K]       learned gating logits.

    Returns ``(gated_prototypes, entropy_loss)`` where ``gated_prototypes`` is
    the soft-masked prototype set and ``entropy_loss`` is a sparsity regularizer
    that encourages a *small* number of active prototypes (adaptive INP count).
    """
    p_on = torch.sigmoid(logits / max(temperature, 1e-6))      # [K]
    gated = prototypes * p_on.view(1, -1, 1)                   # [B, K, C]
    # Entropy over the *expected* activation probabilities pushes p_on towards
    # 0 or 1; combined with an L0-ish penalty it favours fewer active prototypes.
    p = torch.stack([1.0 - p_on, p_on], dim=-1).clamp(min=1e-6)  # [K, 2]
    entropy = -(p * p.log()).sum(dim=-1).mean()
    return gated, entropy


# ---------------------------------------------------------------------------
# Multi-granularity INP-Former
# ---------------------------------------------------------------------------
class INP_Former(nn.Module):
    def __init__(
        self,
        encoder,
        groups: nn.ModuleList,               # per-granularity sub-modules
        prototype_tokens: nn.ParameterList,  # per-granularity INPs [K_g, C]
        target_layers=(2, 3, 4, 5, 6, 7, 8, 9),
        layer_groups=((0, 1, 2, 3, 4, 5, 6, 7),),
        remove_class_token=False,
        encoder_require_grad_layer=(),
        coherence_mode="soft",               # 'soft' | 'hard'
        coherence_temperature=0.1,
        diversity_weight=0.0,
        adaptive_prototypes=False,
        gate_temperature=1.0,
        gate_sparsity_weight=0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.groups = groups
        self.prototype_tokens = prototype_tokens
        self.target_layers = list(target_layers)
        self.layer_groups = [list(g) for g in layer_groups]
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = list(encoder_require_grad_layer)
        self.coherence_mode = coherence_mode
        self.coherence_temperature = coherence_temperature
        self.diversity_weight = diversity_weight
        self.adaptive_prototypes = adaptive_prototypes
        self.gate_temperature = gate_temperature
        self.gate_sparsity_weight = gate_sparsity_weight

        self.num_groups = len(groups)
        assert len(prototype_tokens) == self.num_groups, \
            "one prototype set and one sub-module required per granularity group"
        assert len(self.layer_groups) == self.num_groups, \
            "one layer group required per granularity group"

        # Learned gating logits (one per prototype, per group) for the adaptive
        # prototype-count variant.
        if adaptive_prototypes:
            self.prototype_logits = nn.ParameterList([
                nn.Parameter(torch.zeros(p.shape[0])) for p in prototype_tokens
            ])
        else:
            self.prototype_logits = None

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def fuse_feature(feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def _coherence(self, x: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        if self.coherence_mode == "hard":
            return hard_coherence_loss(x, proto)
        return soft_coherence_loss(x, proto, self.coherence_temperature)

    # -- forward ----------------------------------------------------------
    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        B = x.shape[0]

        # 1. Extract the target intermediate layers (frozen encoder).
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i > self.target_layers[-1]:
                break
            if i in self.encoder_require_grad_layer:
                x = blk(x)
            else:
                with torch.no_grad():
                    x = blk(x)
            if i in self.target_layers:
                en_list.append(x)

        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))
        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        # 2. Per-granularity prototype extraction + reconstruction.
        en_out, de_out = [], []
        coherence_loss = 0.0
        for g in range(self.num_groups):
            # Group input features (indices are positions within target_layers).
            feats = [en_list[i] for i in self.layer_groups[g]]
            x_g = self.fuse_feature(feats)                      # [B, N, C]

            raw_proto = self.prototype_tokens[g]                # [K_g, C]
            group = self.groups[g]

            # (a) INP extractor: aggregate normal tokens into prototypes.
            #     Cross-attention consumes a *batched* proto, so expand the
            #     learnable tokens to [B, K, C] before the loop.
            proto = raw_proto.unsqueeze(0).expand(B, -1, -1)    # [B, K, C]
            for blk in group.aggregation:
                proto = blk(proto, x_g)

            # (b) Optional adaptive gating of the prototype set.
            gate_entropy = x.new_zeros(())
            if self.adaptive_prototypes:
                proto, gate_entropy = adaptive_prototype_gate(
                    proto, self.prototype_logits[g], self.gate_temperature
                )

            # (c) Coherence loss (soft or hard) + optional diversity regularizer.
            coherence_loss = coherence_loss + self._coherence(x_g, proto)
            if self.diversity_weight > 0:
                coherence_loss = coherence_loss + \
                    self.diversity_weight * prototype_diversity_loss(proto)
            if self.gate_sparsity_weight > 0:
                coherence_loss = coherence_loss + \
                    self.gate_sparsity_weight * gate_entropy

            # (d) Bottleneck + INP-guided decoder (per group).
            h = x_g
            for blk in group.bottleneck:
                h = blk(h)
            de_g = []
            for blk in group.decoder:
                h = blk(h, proto)
                de_g.append(h)
            de_g = self.fuse_feature(de_g)                      # [B, N, C]

            # (e) Reshape to [B, C, H, W] feature maps for the anomaly metric.
            en_out.append(x_g.permute(0, 2, 1).reshape(B, -1, side, side).contiguous())
            de_out.append(de_g.permute(0, 2, 1).reshape(B, -1, side, side).contiguous())

        coherence_loss = coherence_loss / self.num_groups
        return en_out, de_out, coherence_loss
