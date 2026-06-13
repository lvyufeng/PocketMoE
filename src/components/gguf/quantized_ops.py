from __future__ import annotations

import torch

from src.kernels.cuda_loader import load_cuda_kernel
from src.loader.gguf.quantized_tensor import QuantizedGGUFTensor


class QuantizedGGUFLinear:
    """CUDA linear over raw GGUF quantized matrix blocks."""

    def __init__(self, tensor: QuantizedGGUFTensor, *, out_dtype: torch.dtype = torch.float16):
        self.tensor = tensor
        self.out_dtype = out_dtype
        self._cuda = load_cuda_kernel()
        if self._cuda is None:
            raise RuntimeError("CUDA extension is required for QuantizedGGUFLinear")
        self._grid = torch.empty(0, dtype=torch.int8, device=tensor.blocks.device)

    @property
    def in_dim(self) -> int:
        return int(self.tensor.row_elems)

    @property
    def out_dim(self) -> int:
        return int(self.tensor.out_dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_dim:
            raise ValueError(f"{self.tensor.source_name}: expected input dim {self.in_dim}, got {x.size(-1)}")
        rows = int(x.numel() // self.in_dim)
        x_contig = x.contiguous()
        if rows > 1:
            y = self._cuda.gguf_quant_gemm_prefill_forward(
                x_contig,
                self.tensor.blocks,
                int(self.tensor.row_elems),
                int(self.tensor.type_id),
                self._grid,
            )
        else:
            y = self._cuda.gguf_quant_gemm_forward(
                x_contig,
                self.tensor.blocks,
                int(self.tensor.row_elems),
                int(self.tensor.type_id),
                self._grid,
            )
        return y.to(self.out_dtype)

    @staticmethod
    def pair(first: "QuantizedGGUFLinear", second: "QuantizedGGUFLinear", x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if first.in_dim != second.in_dim:
            raise ValueError(f"paired GGUF GEMM requires matching input dims, got {first.in_dim} and {second.in_dim}")
        if first.out_dim != second.out_dim:
            raise ValueError(f"paired GGUF GEMM requires matching output dims, got {first.out_dim} and {second.out_dim}")
        if first.tensor.type_id != second.tensor.type_id:
            raise ValueError(f"paired GGUF GEMM requires matching type ids, got {first.tensor.type_id} and {second.tensor.type_id}")
        if x.size(-1) != first.in_dim:
            raise ValueError(f"paired GGUF GEMM expected input dim {first.in_dim}, got {x.size(-1)}")
        y0, y1 = first._cuda.gguf_quant_gemm_pair_forward(
            x.contiguous(),
            first.tensor.blocks,
            int(first.tensor.row_elems),
            int(first.tensor.type_id),
            second.tensor.blocks,
            int(second.tensor.row_elems),
            int(second.tensor.type_id),
            first._grid,
        )
        return y0.to(first.out_dtype), y1.to(second.out_dtype)


class QuantizedGGUFEmbedding:
    """CUDA selected-row embedding over raw GGUF quantized matrix blocks."""

    def __init__(self, tensor: QuantizedGGUFTensor, *, out_dtype: torch.dtype = torch.float16):
        self.tensor = tensor
        self.out_dtype = out_dtype
        self._cuda = load_cuda_kernel()
        if self._cuda is None:
            raise RuntimeError("CUDA extension is required for QuantizedGGUFEmbedding")
        self._grid = torch.empty(0, dtype=torch.int8, device=tensor.blocks.device)

    @property
    def dim(self) -> int:
        return int(self.tensor.row_elems)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.long:
            token_ids = token_ids.to(torch.long)
        y = self._cuda.gguf_quant_embedding_forward(
            token_ids.contiguous(),
            self.tensor.blocks,
            int(self.tensor.row_elems),
            int(self.tensor.type_id),
            self._grid,
        )
        return y.to(self.out_dtype)
