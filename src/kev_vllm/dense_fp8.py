"""Opt-in rowwise Torch FP8 GEMMs for measured Hopper projection shapes.

Uses the existing quantized inputs, weights and per-token/channel scales.
Quantization, all other shapes, and output dtype remain unchanged.
"""
import hashlib
from pathlib import Path
import torch

_SOURCE_SHA256 = "d01b6fe09e3c3fa3991045f6ac7b469a58a550c56426ffd8900badbbe7e5355b"


def install():
    from vllm.model_executor.kernels.linear.scaled_mm import cutlass
    from vllm.platforms import current_platform
    cls = cutlass.CutlassFP8ScaledMMLinearKernel
    if getattr(cls, "_tern_rowwise_gemm", False):
        return
    capability = current_platform.get_device_capability()
    if not current_platform.is_cuda() or capability is None or capability.to_int() != 90:
        raise RuntimeError("The rowwise FP8 experiment requires SM90 CUDA hardware")
    if hashlib.sha256(Path(cutlass.__file__).read_bytes()).hexdigest() != _SOURCE_SHA256:
        raise RuntimeError("Unreviewed CUTLASS adapter source for Torch FP8 experiment")
    original = cls.apply_scaled_mm

    def apply(self, *, A, B, out_dtype, As, Bs, bias, output_shape):
        if (B.shape not in ((2048, 12288), (2048, 1024))
                or A.shape[1] != 2048 or self.logical_output_size != B.shape[1]
                or out_dtype != torch.bfloat16 or A.dtype != torch.float8_e4m3fn
                or B.dtype != torch.float8_e4m3fn or bias is not None
                or As.numel() != A.shape[0] or Bs.numel() != B.shape[1]):
            return original(self, A=A, B=B, out_dtype=out_dtype, As=As, Bs=Bs,
                            bias=bias, output_shape=output_shape)
        output = torch._scaled_mm(A, B, scale_a=As.reshape(-1, 1),
                                  scale_b=Bs.reshape(1, -1),
                                  out_dtype=out_dtype, use_fast_accum=True)
        return output.view(*output_shape[:-1], self.logical_output_size)

    cls.apply_scaled_mm = apply
    cls._tern_rowwise_gemm = True
