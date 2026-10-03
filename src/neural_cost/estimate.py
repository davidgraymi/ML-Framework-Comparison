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


def estimate_conv2d(
    data: tuple[int, ...] | Operation,
    kernel: tuple[int, ...] | None = None,
    output: tuple[int, ...] | None = None,
    groups: int = 1,
) -> int:
    """Calculate FLOPs for 2D convolution with grouped and depthwise support.

    Accepts either an Operation instance or explicit (data, kernel, output, groups) shapes.
    Enforces the channel invariant: data[1] == kernel[1] * groups.
    """
    if isinstance(data, Operation):
        op = data
        if len(op.inputs) < 2:
            raise ValueError("conv2d requires input and kernel shapes")
        data_shape, kernel_shape = op.inputs[:2]
        output_shape = op.output
        groups_val = int(op.attrs.get("groups", groups))
    else:
        if kernel is None or output is None:
            raise ValueError("explicit shapes require data, kernel, and output")
        data_shape = data
        kernel_shape = kernel
        output_shape = output
        groups_val = groups

    if len(data_shape) != 4 or len(kernel_shape) != 4 or len(output_shape) != 4:
        raise ValueError("conv2d expects NCHW input, OIHW kernel, and NCHW output")
    if groups_val <= 0:
        raise ValueError("groups must be a positive integer")

    if data_shape[1] != kernel_shape[1] * groups_val or output_shape[1] != kernel_shape[0]:
        raise ValueError("conv2d channel dimensions do not match")

    assert data_shape[1] == kernel_shape[1] * groups_val

    return 2 * numel(output_shape) * kernel_shape[1] * kernel_shape[2] * kernel_shape[3]


def estimate_operation(operation: Operation) -> CostEstimate:
    """Estimate FLOPs and compulsory tensor I/O for one operation.

    The estimate intentionally models useful work and compulsory traffic, not
    cache misses, workspace, fusion, or allocator effects.  Those differences
    become visible in :func:`neural_cost.analyze_gap`.
    """
    read_bytes = sum(numel(shape) for shape in operation.inputs) * operation.dtype_bytes + int(
        operation.attrs.get("parameter_bytes", 0)
    )
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
        flops = estimate_conv2d(operation)
    elif kind == "elementwise":
        flops = int(operation.attrs.get("flops_per_element", 1) * numel(operation.output))
    elif kind == "custom":
        if "flops" not in operation.attrs:
            raise ValueError("custom operations require attrs['flops']")
        flops = int(operation.attrs["flops"])
    elif kind == "softmax" or kind in {"layernorm", "batchnorm"}:
        flops = 5 * numel(operation.output)
    elif kind == "rmsnorm":
        flops = 3 * numel(operation.output)
    elif kind == "swiglu":
        flops = int(
            operation.attrs.get(
                "flops", operation.attrs.get("flops_per_element", 3) * numel(operation.output)
            )
        )
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
        if len(operation.output) >= 4:
            b = operation.output[0]
            num_heads = int(operation.attrs.get("num_heads", operation.output[1]))
            seq_len = int(operation.attrs.get("seq_len", operation.output[2]))
            head_dim = int(operation.attrs.get("head_dim", operation.output[3]))
            d = num_heads * head_dim
            b_t = b * seq_len
        elif len(operation.output) >= 2:
            b_t = prod(operation.output[:-1])  # batch * seq combined
            d = operation.output[-1]
            num_heads = int(operation.attrs.get("num_heads", 1))
            head_dim = int(operation.attrs.get("head_dim", d // max(num_heads, 1)))
            seq_len = int(
                operation.attrs.get(
                    "seq_len", operation.output[-2] if len(operation.output) >= 2 else 1
                )
            )
        else:
            b_t, d, num_heads, head_dim, seq_len = (
                1,
                numel(operation.output),
                1,
                numel(operation.output),
                1,
            )

        include_projections = bool(operation.attrs.get("include_projections", True))
        proj_flops = 4 * 2 * b_t * d * d if include_projections else 0
        attn_flops = 4 * b_t * num_heads * seq_len * head_dim
        softmax_flops = 5 * b_t * num_heads * seq_len
        flops = proj_flops + attn_flops + softmax_flops
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


@dataclass(frozen=True, slots=True)
class FusedCostEstimate:
    """Cost estimate accounting for compiler operator fusion."""

    unfused: CostEstimate
    fused_read_bytes: int
    fused_write_bytes: int
    eliminated_bytes: int
    fused_groups_count: int

    @property
    def flops(self) -> int:
        return self.unfused.flops

    @property
    def read_bytes(self) -> int:
        return self.fused_read_bytes

    @property
    def write_bytes(self) -> int:
        return self.fused_write_bytes

    @property
    def operations(self) -> int:
        return self.unfused.operations

    @property
    def total_bytes(self) -> int:
        return self.fused_read_bytes + self.fused_write_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.total_bytes if self.total_bytes else 0.0

    @property
    def traffic_reduction_ratio(self) -> float:
        return self.eliminated_bytes / self.unfused.total_bytes if self.unfused.total_bytes else 0.0


_FUSIBLE_CONSUMER_KINDS = {
    "elementwise",
    "softmax",
    "layernorm",
    "rmsnorm",
    "batchnorm",
    "pooling",
    "swiglu",
}


def estimate_fused_operations(operations: Iterable[Operation]) -> FusedCostEstimate:
    """Estimate theoretical work and reduced tensor traffic under operator fusion.

    Identifies producer-consumer patterns (such as linear/conv2d followed by
    elementwise, normalization, or pooling layers) that modern optimizing
    compilers (e.g., PyTorch Inductor, JAX/XLA) fuse into single kernels,
    eliminating intermediate DRAM roundtrips.
    """
    op_list = list(operations)
    unfused = estimate_operations(op_list)
    if not op_list:
        return FusedCostEstimate(unfused, 0, 0, 0, 0)

    eliminated_read_bytes = 0
    eliminated_write_bytes = 0
    fused_groups = 0

    i = 0
    while i < len(op_list):
        current_op = op_list[i]
        fused_with_current = False
        j = i + 1
        last_out_shape = current_op.output
        last_dtype = current_op.dtype_bytes

        while j < len(op_list):
            next_op = op_list[j]
            if (
                next_op.kind in _FUSIBLE_CONSUMER_KINDS
                and next_op.inputs
                and next_op.inputs[0] == last_out_shape
            ):
                intermediate_bytes = numel(last_out_shape) * min(last_dtype, next_op.dtype_bytes)
                eliminated_write_bytes += intermediate_bytes
                eliminated_read_bytes += intermediate_bytes
                fused_with_current = True
                last_out_shape = next_op.output
                last_dtype = next_op.dtype_bytes
                j += 1
            else:
                break

        if fused_with_current:
            fused_groups += 1
            i = j
        else:
            i += 1

    fused_read = max(0, unfused.read_bytes - eliminated_read_bytes)
    fused_write = max(0, unfused.write_bytes - eliminated_write_bytes)
    eliminated_total = eliminated_read_bytes + eliminated_write_bytes

    return FusedCostEstimate(
        unfused=unfused,
        fused_read_bytes=fused_read,
        fused_write_bytes=fused_write,
        eliminated_bytes=eliminated_total,
        fused_groups_count=fused_groups,
    )


def estimate_adamw_traffic(num_parameters: int, dtype_bytes: int = 4) -> int:
    """Model DRAM traffic for an AdamW optimizer step across model parameters.

    For parameter count P:
      Reads:  Parameter (P), Gradient (P), First Moment (P), Second Moment (P) = 4P
      Writes: Updated Parameter (P), First Moment (P), Second Moment (P)       = 3P
      Total DRAM bytes: (4P + 3P) * dtype_bytes = 7 * P * dtype_bytes.
    """
    return 7 * int(num_parameters) * int(dtype_bytes)
