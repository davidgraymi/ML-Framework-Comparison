"""Theoretical operation cost estimation."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import prod

from .operations import Operation, numel


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Theoretical work and minimum tensor traffic for a workload."""

    flops: int
    read_bytes: int
    write_bytes: int
    operations: int

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.total_bytes if self.total_bytes else 0.0

    def __add__(self, other: "CostEstimate") -> "CostEstimate":
        return CostEstimate(
            self.flops + other.flops,
            self.read_bytes + other.read_bytes,
            self.write_bytes + other.write_bytes,
            self.operations + other.operations,
        )


def estimate_operation(operation: Operation) -> CostEstimate:
    """Estimate FLOPs and compulsory tensor I/O for one operation.

    The estimate intentionally models useful work and compulsory traffic, not
    cache misses, workspace, fusion, or allocator effects.  Those differences
    become visible in :func:`neural_cost.analyze_gap`.
    """
    read_bytes = sum(numel(shape) for shape in operation.inputs) * operation.dtype_bytes
    write_bytes = numel(operation.output) * operation.dtype_bytes
    kind = operation.kind

    if kind in {"matmul", "linear"}:
        if len(operation.inputs) < 2:
            raise ValueError(f"{kind} requires input and weight shapes")
        left, right = operation.inputs[:2]
        if len(left) < 2 or len(right) != 2:
            raise ValueError(f"{kind} expects [..., M, K] and [K, N] shapes")
        if left[-1] != right[0]:
            raise ValueError("matrix inner dimensions do not match")
        flops = 2 * prod(left[:-1]) * right[0] * right[1]
    elif kind == "conv2d":
        if len(operation.inputs) < 2:
            raise ValueError("conv2d requires input and kernel shapes")
        data, kernel = operation.inputs[:2]
        if len(data) != 4 or len(kernel) != 4 or len(operation.output) != 4:
            raise ValueError("conv2d expects NCHW input, OIHW kernel, and NCHW output")
        if data[1] != kernel[1] or operation.output[1] != kernel[0]:
            raise ValueError("conv2d channel dimensions do not match")
        flops = 2 * numel(operation.output) * kernel[1] * kernel[2] * kernel[3]
    elif kind == "elementwise":
        flops = int(operation.attrs.get("flops_per_element", 1) * numel(operation.output))
    elif kind == "custom":
        if "flops" not in operation.attrs:
            raise ValueError("custom operations require attrs['flops']")
        flops = int(operation.attrs["flops"])
    elif kind == "softmax" or kind in {"layernorm", "batchnorm"}:
        flops = 5 * numel(operation.output)
    elif kind == "embedding":
        # Embedding lookup: no multiply-accumulate FLOPs, just one read per token.
        # output shape is (batch, seq_len, embed_dim) or (N, embed_dim).
        flops = numel(operation.output)  # 1 FLOP/element as a nominal indexing cost
    elif kind == "attention":
        # Multi-head self-attention cost:
        #   4 linear projections  : 4 × 2 × B × T × D² / num_heads × num_heads = 4×2×B×T×D²
        #   QKᵀ per head         : B × H × T × T × head_dim  (2 FLOPs per element)
        #   softmax               : 5 × B × H × T²
        #   weighted sum (AV)     : B × H × T × T × head_dim (2 FLOPs)
        # We approximate using output shape (B, T, D) and attrs.
        if len(operation.output) >= 2:
            b_t = prod(operation.output[:-1])  # batch * seq combined
            d = operation.output[-1]
            num_heads = int(operation.attrs.get("num_heads", 1))
            head_dim = d // max(num_heads, 1)
            seq_len = int(operation.attrs.get("seq_len", operation.output[-2] if len(operation.output) >= 2 else 1))
            # 4 linear projections (Q, K, V, O)
            proj_flops = 4 * 2 * b_t * d * d
            # QKᵀ + AV per head  (2 passes over T×T×head_dim each)
            attn_flops = 4 * b_t * num_heads * seq_len * head_dim
            # softmax over seq_len per head (5 ops)
            softmax_flops = 5 * b_t * num_heads * seq_len
            flops = proj_flops + attn_flops + softmax_flops
        else:
            flops = numel(operation.output)
    elif kind == "pooling":
        kernel_size = operation.attrs.get("kernel_size")
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise ValueError("pooling requires attrs['kernel_size'] as (kH, kW)")
        flops = numel(operation.output) * kernel_size[0] * kernel_size[1]
    else:  # Defensive in case a caller bypasses static typing.
        raise ValueError(f"unsupported operation kind: {kind}")
    return CostEstimate(flops, read_bytes, write_bytes, 1)


def estimate_operations(operations: Iterable[Operation]) -> CostEstimate:
    """Aggregate theoretical cost for an iterable of operations."""
    total = CostEstimate(0, 0, 0, 0)
    for operation in operations:
        total += estimate_operation(operation)
    return total
