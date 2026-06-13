from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizedGGUFTensor:
    """Device-resident raw GGUF quantized matrix blocks.

    The tensor stores raw GGUF blocks, not dequantized weights.  ``row_start`` is
    non-zero for row-sliced tensors such as TP-sharded vocab/lm_head weights.
    """

    source_name: str
    blocks: torch.Tensor
    type_name: str
    type_id: int
    row_elems: int
    out_dim: int
    row_start: int = 0

    @property
    def nbytes(self) -> int:
        return int(self.blocks.numel() * self.blocks.element_size())
