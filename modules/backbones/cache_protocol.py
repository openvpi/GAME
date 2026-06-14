"""CachableBackbone protocol mixin for DBCache step caching.

Any backbone that implements this protocol can be used with
:class:`inference.cache.DBCacheSegmenter`.  The protocol decouples the
caching logic from concrete backbone internals (``EBFBackbone``, etc.).

See Also
--------
CACHE_DIT.md : Algorithm description and benchmark results.
inference.cache.DBCacheSegmenter : The cache implementation.
"""
from __future__ import annotations

import torch

__all__ = ["CachableBackbone"]


class CachableBackbone:
    """Mixin for backbones whose forward pass is a sequential block stack.

    A cacheable backbone has the following structure::

        x -> input_head -> [block_0, block_1, ..., block_{N-1}] -> output_head -> out

    An optional *latent tap* at an intermediate block extracts a lower
    dimensional representation for auxiliary tasks (e.g. self-similarity).

    :meth:`input_head` and :meth:`output_head` have identity defaults and
    may be omitted.  All other methods **must** be overridden.
    """

    # ---- properties -------------------------------------------------------

    @property
    def num_blocks(self) -> int:
        """Number of sequential blocks in the backbone."""
        raise NotImplementedError

    @property
    def latent_block_idx(self) -> int | None:
        """Zero-indexed block at which to extract the latent, or ``None``."""
        raise NotImplementedError

    @property
    def returns_latent(self) -> bool:
        """``True`` when :meth:`forward` returns ``(out, latent)``."""
        return self.latent_block_idx is not None

    # ---- head methods -----------------------------------------------------

    def input_head(self, x: torch.Tensor) -> torch.Tensor:
        """Preliminary projection applied before the block stack.

        Default: identity (no-op).  Override when a learned projection
        (e.g. ``nn.Linear``) is needed.
        """
        return x

    def output_head(self, x: torch.Tensor) -> torch.Tensor:
        """Final norm and projection applied after the block stack.

        Default: identity (no-op).  Override when an output norm and/or
        projection (e.g. ``RMSNorm`` + ``nn.Linear``) is needed.
        """
        return x

    # ---- block execution --------------------------------------------------

    def run_front(
        self,
        x: torch.Tensor,
        n: int,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the first *n* blocks (indices ``0 .. n-1``).

        :returns: Intermediate activations after block ``n-1``.
        """
        raise NotImplementedError

    def run_tail(
        self,
        x: torch.Tensor,
        start: int,
        *,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run blocks from *start* to the end (indices ``start .. num_blocks-1``).

        :returns: ``(output_after_last_block, latent_or_None)``.
            If the latent tap falls inside the tail region, the latent is
            extracted and returned.  It is the caller's responsibility to
            ensure ``start <= latent_block_idx`` (or that no latent is
            needed) — otherwise a ``RuntimeError`` is raised.
        """
        raise NotImplementedError

    # ---- latent extraction ------------------------------------------------

    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the latent norm and projection at the tap point."""
        raise NotImplementedError
