import unittest
from typing import get_args

from neural_cost.estimate import estimate_operation
from neural_cost.operations import Operation, OperationKind


class TestOperationsExpanded(unittest.TestCase):
    def test_softmax_flops(self):
        op = Operation(
            name="softmax_op",
            kind="softmax",
            inputs=((2, 3, 4),),
            output=(2, 3, 4),
            dtype_bytes=4,
        )
        est = estimate_operation(op)
        # 5 * numel(output)
        self.assertEqual(est.flops, 5 * 24)

    def test_layernorm_flops(self):
        op = Operation(
            name="ln_op",
            kind="layernorm",
            inputs=((2, 3, 4), (4,), (4,)),
            output=(2, 3, 4),
            dtype_bytes=4,
        )
        est = estimate_operation(op)
        self.assertEqual(est.flops, 5 * 24)

    def test_batchnorm_flops(self):
        op = Operation(
            name="bn_op",
            kind="batchnorm",
            inputs=((2, 4, 3, 3), (4,), (4,), (4,), (4,)),
            output=(2, 4, 3, 3),
            dtype_bytes=4,
        )
        est = estimate_operation(op)
        self.assertEqual(est.flops, 5 * 72)

    def test_pooling_flops(self):
        op = Operation(
            name="pool_op",
            kind="pooling",
            inputs=((1, 3, 10, 10),),
            output=(1, 3, 5, 5),
            dtype_bytes=2,
            attrs={"kernel_size": (2, 2)}
        )
        est = estimate_operation(op)
        # numel(output) * kH * kW = 75 * 2 * 2 = 300
        self.assertEqual(est.flops, 300)

    def test_pooling_missing_kernel_raises(self):
        op = Operation(
            name="pool_op",
            kind="pooling",
            inputs=((1, 3, 10, 10),),
            output=(1, 3, 5, 5),
            dtype_bytes=2,
            # missing attrs
        )
        with self.assertRaisesRegex(ValueError, r"pooling requires attrs\['kernel_size'\] as \(kH, kW\)"):
            estimate_operation(op)

    def test_all_new_kinds_in_operation_kind(self):
        valid_kinds = get_args(OperationKind)
        self.assertIn("softmax", valid_kinds)
        self.assertIn("layernorm", valid_kinds)
        self.assertIn("batchnorm", valid_kinds)
        self.assertIn("pooling", valid_kinds)

    def test_read_write_bytes_for_new_ops(self):
        # Softmax
        op = Operation("op1", "softmax", ((10,),), (10,), 4)
        est = estimate_operation(op)
        self.assertEqual(est.read_bytes, 40)
        self.assertEqual(est.write_bytes, 40)
        
        # LayerNorm
        op = Operation("op2", "layernorm", ((10,), (10,)), (10,), 2)
        est = estimate_operation(op)
        self.assertEqual(est.read_bytes, 40) # (10 + 10) * 2
        self.assertEqual(est.write_bytes, 20) # 10 * 2
        
if __name__ == '__main__':
    unittest.main()
