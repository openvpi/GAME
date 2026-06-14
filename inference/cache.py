"""DBCache-style step caching for the GAME segmenter.

Inspired by [cache-dit](https://github.com/vipshop/cache-dit)'s DBCache. The
segmenter (an N-layer EBF transformer-conv hybrid) is invoked once per D3PM
sampling step (`nsteps` total). Adjacent steps produce highly similar block
outputs, so we cache the residual `(x_tail - x_front)` from the previous step
and reuse it when the front-block residual delta falls below a threshold.

Algorithm
---------
For each forward pass of the segmenter:

1. Run only the first ``Fn`` blocks → ``x_front``.
2. Compute ``delta = mean|x_front - x_front_prev| / mean|x_front_prev|``.
3. If ``step >= warmup`` and ``delta < threshold`` and a cached tail delta
   exists: skip the remaining blocks and reuse the cached delta:
   ``x_out = x_front + tail_delta`` (and reuse the cached intermediate latent).
4. Otherwise: run the full block stack, refresh ``tail_delta`` and the latent
   cache.

Only the **segmenter** is wrapped — the estimator is invoked once per audio
batch (not iteratively) so caching brings nothing there. Pitch quality is not
affected because the estimator path is untouched.

Reset semantics: ``DBCacheSegmenter.reset()`` clears the per-segment state.
``hits`` / ``misses`` are aggregate counters across all resets and are useful
for reporting hit rates over a full run.

Example
-------

.. code-block:: python

    from inference.api import load_inference_model
    from inference.cache import DBCacheSegmenter

    model, lang_map = load_inference_model(ckpt_path)
    cacher = DBCacheSegmenter(
        model.model.segmenter, fn_blocks=1, threshold=0.25, warmup_steps=1
    ).install_into(model)

    # ... run inference as usual ...
    print(f"cache hit rate: {cacher.hit_rate:.1%}")
"""
from __future__ import annotations

import torch
from torch import nn

from modules.backbones.cache_protocol import CachableBackbone

__all__ = ["DBCacheSegmenter"]


class DBCacheSegmenter:
    """Residual-difference step cache for the segmenter.

    :param segmenter: The segmenter module (an ``EBFBackbone`` instance). Its
        ``forward`` is monkey-patched to introduce caching.
    :param fn_blocks: Number of leading blocks always executed (the "front"
        used to compute the residual delta). 1 is usually enough.
    :param threshold: Maximum normalized L1 delta below which the tail blocks
        are skipped. Higher values → more cache hits → faster, slightly less
        accurate. Typical range 0.08–0.40.
    :param warmup_steps: Number of full forward passes before any cache hit is
        allowed. The first step has no previous state to compare against, so
        ``warmup_steps=1`` is the minimum sensible value.
    """

    def __init__(
        self,
        segmenter: nn.Module,
        fn_blocks: int = 1,
        threshold: float = 0.25,
        warmup_steps: int = 1,
    ):
        if fn_blocks < 1:
            raise ValueError("fn_blocks must be >= 1")
        if not isinstance(segmenter, CachableBackbone):
            raise TypeError(
                f"segmenter must be a CachableBackbone instance, "
                f"got {type(segmenter).__name__}"
            )
        if (
            segmenter.latent_block_idx is not None
            and segmenter.latent_block_idx < fn_blocks
        ):
            raise ValueError(
                f"latent_block_idx ({segmenter.latent_block_idx}) must be "
                f">= fn_blocks ({fn_blocks})."
            )
        self.seg = segmenter
        self.fn = fn_blocks
        self.threshold = float(threshold)
        self.warmup = int(warmup_steps)
        self.hits = 0
        self.misses = 0
        self._orig_forward = None
        self.reset()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def reset(self) -> None:
        """Clear per-segment state. Aggregate hit counters are preserved."""
        self._step = 0
        self._prev_front: torch.Tensor | None = None
        self._tail_delta: torch.Tensor | None = None
        self._latent_cache: torch.Tensor | None = None

    def install_into(self, inference_model) -> "DBCacheSegmenter":
        """Install the cache and hook ``reset()`` into the inference model.

        The reset hook fires at the start of every ``forward_segmenter_main``
        call, which corresponds to one audio segment in the batch. This means
        the cache is reused across the ``nsteps`` D3PM iterations of a single
        segment but is never carried across unrelated segments.
        """
        self._install_segmenter_hook()
        orig_fsm = inference_model.forward_segmenter_main
        cacher = self

        def fsm_with_reset(*args, **kwargs):
            cacher.reset()
            return orig_fsm(*args, **kwargs)

        inference_model.forward_segmenter_main = fsm_with_reset
        return self

    def _install_segmenter_hook(self) -> None:
        self._orig_forward = self.seg.forward
        cacher = self

        def cached_forward(x: torch.Tensor, mask: torch.Tensor | None = None):
            seg = cacher.seg
            x = seg.input_head(x)

            # Always run the front blocks.
            x_front = seg.run_front(x, cacher.fn, mask=mask)

            # Decide whether to skip the tail blocks.
            use_cache = False
            if (
                cacher._prev_front is not None
                and cacher._step >= cacher.warmup
                and cacher._tail_delta is not None
            ):
                num = (x_front - cacher._prev_front).abs().mean()
                den = cacher._prev_front.abs().mean() + 1e-8
                delta = (num / den).item()
                if delta < cacher.threshold:
                    use_cache = True

            if use_cache:
                x_out = x_front + cacher._tail_delta
                latent = cacher._latent_cache
                cacher.hits += 1
            else:
                x_run, latent = seg.run_tail(x_front, cacher.fn, mask=mask)
                x_out = x_run
                cacher._tail_delta = (x_out - x_front).detach()
                cacher._latent_cache = (
                    latent.detach() if latent is not None else None
                )
                cacher.misses += 1

            cacher._prev_front = x_front.detach()
            cacher._step += 1

            out = seg.output_head(x_out)
            if seg.returns_latent:
                return out, latent
            return out

        self.seg.forward = cached_forward

    def uninstall(self) -> None:
        """Restore the original segmenter forward and clear cache state."""
        if self._orig_forward is not None:
            self.seg.forward = self._orig_forward
            self._orig_forward = None
        self.reset()
