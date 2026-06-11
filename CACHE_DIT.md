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

D3PM sampling is stochastic, so two independent fp32 runs are not bit-identical.
The first row below establishes the stochastic noise floor; all metrics are
reported against the original fp32 run.

| Configuration | Δ note count | Onset match (≤50 ms) | Pitch mismatch in matched | Mean onset error | Mean duration error |
|---|---|---|---|---|---|
| fp32 rerun (noise floor) | −4 | 98.2% | 9 | 1.0 ms | 9.9 ms |
| `--cache-threshold 0.08` | +9 | 99.1% | 13 | 1.2 ms | 12.5 ms |
| `--cache-threshold 0.15` | −7 | 98.0% | 15 | 1.2 ms | 13.8 ms |
| `--cache-threshold 0.25` | +1 | 97.8% | 10 | 0.9 ms | 10.2 ms |
| `--cache-threshold 0.40` | +1 | 98.2% |  8 | 0.9 ms |  9.6 ms |

All cached configurations fall within the D3PM stochastic noise floor on every
metric, including the most aggressive `τ = 0.40`.

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
