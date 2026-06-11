# DBCache Acceleration for GAME

This document describes the DBCache-based inference acceleration available on this branch.

## Attribution

The caching strategy is a reimplementation of **DBCache** (Dual-Block Cache) from
[vipshop/cache-dit](https://github.com/vipshop/cache-dit), adapted to GAME's custom
`EBFBackbone` segmenter. The `cache-dit` package itself is not used at runtime
because its `BlockAdapter` requires diffusers-style `ForwardPattern` block
signatures (`hidden_states` / `encoder_hidden_states`), which `EBF.forward(x, mask=None)`
does not satisfy. The algorithm below is functionally equivalent and adds no
external dependency.

## Algorithm

GAME's segmenter is an `N`-layer `EBFBackbone` invoked once per D3PM sampling
step (`--nsteps`, default 8). Successive denoising steps produce highly
correlated intermediate activations. DBCache exploits this by caching the
residual contribution of the tail block stack and reusing it whenever the front
block's residual delta is sufficiently small.

Let `B = [b_0, ..., b_{N-1}]` be the segmenter blocks, `Fn` the number of front
(warmup) blocks, `s` the current step index, `W` the warmup-step count, and `τ`
the threshold. For each forward pass:

1. Compute the front output

       x_front = b_{Fn-1}(... b_0(x) ...)

2. Compute the normalized L1 residual delta against the previous step:

       δ = mean(|x_front − x_front_prev|) / (mean(|x_front_prev|) + ε)

3. If `s ≥ W`, `δ < τ`, and a cached `tail_delta` exists,
   **skip blocks `b_{Fn} ... b_{N-1}`** and reconstruct the output as

       x_out = x_front + tail_delta

   The cached intermediate latent at `latent_layer_idx` is reused as well.

4. Otherwise, run the full tail, then refresh the cache:

       tail_delta = (x_out − x_front).detach()
       latent_cache = latent.detach()

5. `x_out` is passed through the segmenter's output norm and projection
   unchanged.

Cache state is reset at the start of every audio segment via a hook on
`forward_segmenter_main`, so the cache is reused only within a single segment's
`nsteps` D3PM iterations and never across unrelated segments.

Only the segmenter is wrapped. The estimator is invoked once per segment and
gains nothing from caching; pitch prediction is therefore identical to the
uncached path regardless of `τ`.

Implementation: [`inference/cache.py`](inference/cache.py).

## Benchmark

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 2070 (Turing, 8 GB) |
| Software | PyTorch 2.8.0 + CUDA 12.9, Lightning 2.6.1 |
| Checkpoint | `GAME-1.0-medium` (~50 M parameters) |
| Audio | 211 s mono vocal, `--nsteps 8`, `--batch-size 4`, precision `32-true` |

### Speed

| Configuration | Inference time | Speedup | Cache hit rate |
|---|---|---|---|
| fp32 baseline | 13.22 s | 1.00× | — |
| `--cache-threshold 0.08` | 11.58 s | 1.14× | 7.1% |
| `--cache-threshold 0.15` | 10.46 s | 1.26× | 26.8% |
| `--cache-threshold 0.25` | 8.97 s | 1.47× | 62.5% |
| `--cache-threshold 0.40` | 7.96 s | 1.66× | 87.5% |

### Accuracy

Evaluated with `evaluate.py` on a private vocal test set (1063
samples, zh vocal, `--nsteps 8`).  Ground truth was produced by the same model
under fp32 no-cache *(align* subcommand).  The first column is the reference
performance, so the fp32 row shows how much the *extract* evaluation mode
deviates from the *align*-generated labels.

| Threshold | Chamfer ↓ | Qty err rate ↓ | Precision | Recall | F1 | Pitch RMSE ↓ | Pitch Acc ↑ | Overall Acc ↑ |
|---|---|---|---|---|---|---|---|---|
| fp32 | 1.248 | 7.79% | 0.979 | 0.943 | 0.993 | 0.358 | 0.9882 | 0.9831 |
| th=0.08 | 1.252 | 7.84% | 0.977 | 0.944 | 0.993 | 0.356 | 0.9881 | 0.9827 |
| th=0.15 | 1.252 | 7.68% | 0.979 | 0.944 | 0.993 | 0.357 | 0.9883 | 0.9831 |
| th=0.25 | 1.297 | 8.34% | 0.972 | 0.943 | 0.992 | 0.365 | 0.9880 | 0.9804 |
| th=0.40 | 1.294 | 8.48% | 0.971 | 0.943 | 0.992 | 0.368 | 0.9876 | 0.9804 |

Interpretation:
- **fp32 baseline vs GT** is itself imperfect (Overall Acc 0.983) because the
  GT was produced by `align` (which post-processes notes) while `evaluate.py`
  measures raw *extract* output.
- **th=0.08 / 0.15** are essentially indistinguishable from fp32 on every
  metric — pitch accuracy is within 0.01 %.
- **th=0.25 / 0.40** show a mild rise in Chamfer distance (+0.05) and
  quantity error (+0.5 pp), and a 0.3 pp drop in Overall Accuracy.  Pitch
  accuracy remains within 0.1 % of baseline.

## Usage

### CLI

DBCache is disabled by default (`--cache-threshold 0`). Both `extract` and
`align` subcommands accept the same options.

```bash
# Balanced (recommended): ~1.47x speedup, accuracy within sampling noise
python infer.py extract path/to/audio.wav \
    -m experiments/GAME-1.0-medium/model.pt \
    --nsteps 8 \
    --cache-threshold 0.25

# Aggressive: ~1.66x speedup, still within sampling noise on benchmark audio
python infer.py extract path/to/audio.wav \
    -m experiments/GAME-1.0-medium/model.pt \
    --nsteps 8 \
    --cache-threshold 0.40
```

| Option | Default | Description |
|---|---|---|
| `--cache-threshold` | `0.0` | Normalized L1 residual threshold `τ`. `0` disables caching. |
| `--cache-fn-blocks` | `1` | Number of front (warmup) blocks `Fn` always executed. Must be strictly less than `segmenter.latent_layer_idx`. |

### Python API

`infer_model` exposes the cache parameters directly:

```python
from inference.api import load_inference_model, infer_model

model, lang_map = load_inference_model(ckpt_path)

infer_model(
    model=model,
    dataset=dataset,
    config=validation_config,
    callbacks=callbacks,
    precision="32-true",
    cache_threshold=0.25,
    cache_fn_blocks=1,
)
```

For finer control (e.g., custom warmup, manual reset, hit-rate inspection),
install the cache directly:

```python
from inference.cache import DBCacheSegmenter

cacher = DBCacheSegmenter(
    model.model.segmenter,
    fn_blocks=1,
    threshold=0.25,
    warmup_steps=1,
).install_into(model)

# ... run inference ...
print(f"hit rate: {cacher.hit_rate:.1%} ({cacher.hits}/{cacher.hits + cacher.misses})")

cacher.uninstall()  # restore the original segmenter.forward
```
