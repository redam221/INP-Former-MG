# INP-Former-MG — Technical Details

This document explains, at code level, exactly what was changed relative to the
original INP-Former and why. Read this before diving into the implementation.

---

## 1. Original INP-Former data flow (recap)

The original `INP_Former.forward` (see the upstream repo) does:

```python
# 1. extract 8 intermediate layers from a frozen DINOv2 encoder
en_list = [block_i(x) for i in target_layers]          # 8 x [B, N, C]

# 2. collapse all 8 layers into ONE fused feature
x = fuse_feature(en_list)                              # mean -> [B, N, C]

# 3. aggregate ONE set of K prototypes from x
agg_proto = prototype_token                            # [K, C] (K = INP_num = 6)
for blk in aggregation:                                # 1 x Aggregation_Block
    agg_proto = blk(agg_proto, x)

# 4. coherence loss: hard nearest-prototype assignment
g_loss = gather_loss(x, agg_proto)                     # torch.min over cos dist

# 5. bottleneck + single decoder, all 8 blocks share agg_proto
x = bottleneck(x)
de_list = [decoder_blk(x, agg_proto) for blk in decoder]   # 8 x Prototype_Block

# 6. reconstruct error = cosine distance(encoder, decoder) per fused scale
```

The key observation: **steps 2–5 share a single prototype set across all
depths.**

---

## 2. What we changed

### 2.1 Per-granularity prototype sets (the core change)

Instead of one fused feature and one prototype set, the intermediate layers are
partitioned into `G` groups via `layer_groups`, and each group owns a full
prototype pipeline:

```
for g in range(G):
    feats    = [en_list[i] for i in layer_groups[g]]   # this group's layers
    x_g      = fuse_feature(feats)                     # [B, N, C]
    proto_g  = extractor_g(INP_g, x_g)                 # this group's own INPs
    de_g     = decoder_g(bottleneck_g(x_g), proto_g)   # this group's own decoder
```

- `INP_g` is a learnable `[K_g, C]` parameter per group (`prototype_tokens`).
- `extractor_g`, `bottleneck_g`, `decoder_g` live in `groups[g]`.
- `G == 1` ⇒ `layer_groups = [[0..7]]` ⇒ identical to the original.

The per-group anomaly maps (`1 − cos(en_g, de_g)`) are averaged by the existing
`cal_anomaly_maps`, so downstream evaluation is unchanged.

### 2.2 Soft-assignment coherence loss

Original (`gather_loss`):

```
dist   = 1 − cosine(x, proto)                # [B, N, K]
g_loss = mean(min_k(dist))                   # hard assignment
```

The `min` zeroes the gradient to every prototype except the nearest one, so a
prototype that is *almost* nearest to many tokens receives no training signal
and can drift.

Ours (`soft_coherence_loss`):

```
sim    = cosine(x, proto)                    # [B, N, K]
w      = softmax(sim / T)                    # soft assignment, temperature T
g_loss = mean( Σ_k w · (1 − sim) )           # soft-weighted distance
```

Every prototype now receives gradient proportional to its (soft) responsibility
for each token, which keeps the whole set covering the normal manifold.

### 2.3 Prototype diversity regularization

```
diversity = mean_{i≠j} max(0, cosine(proto_i, proto_j))
```

A positive off-diagonal cosine similarity means two INPs encode the same normal
mode (collapse). Penalising it pushes the prototypes apart. Added to the
coherence loss with weight `diversity_weight`.

### 2.4 Adaptive prototype gating

Each group has learned logits `z ∈ R^{K_g}`. The prototypes are soft-masked:

```
p_on   = sigmoid(z / T_gate)
proto' = proto · p_on
```

An entropy term plus an optional sparsity weight encourages `p_on` to be
binary and *sparse*, so redundant prototypes are switched off — replacing the
hand-tuned global `INP_num` with a per-group learned effective count.

---

## 3. Parameter budget

For a fair comparison, `decoder_layers` defaults to `len(layer_groups[g])`, so
the total decoder depth across all groups equals the original 8 blocks. The
number of INPs is split across groups (e.g. `6` → `4,4,4` for 3 groups) rather
than multiplied, keeping the prototype parameter count comparable.

| Config | Groups | INPs | Decoder blocks |
|---|---|---|---|
| baseline (original) | 1 | 6 | 8 |
| 2-granularity | 2 | 6+6 | 4+4 = 8 |
| 3-granularity | 3 | 4+4+4 | 3+3+2 = 8 |

---

## 4. Code map

| Concern | File | Symbol |
|---|---|---|
| Multi-granularity forward | `models/uad.py` | `INP_Former` |
| Soft coherence | `models/uad.py` | `soft_coherence_loss` |
| Hard coherence (original) | `models/uad.py` | `hard_coherence_loss` |
| Diversity regularization | `models/uad.py` | `prototype_diversity_loss` |
| Adaptive gating | `models/uad.py` | `adaptive_prototype_gate` |
| Model construction | `models/builder.py` | `build_model` |
| Cross-attention blocks | `models/vision_transformer.py` | `Aggregation_Block`, `Prototype_Block` (unchanged) |
| Training / evaluation | `train.py` | — |
| Logic smoke test | `smoke_test.py` | — |
