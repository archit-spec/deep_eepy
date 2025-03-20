from deep_gemm import gemm_fp8_fp8_bf16_nt
import torch
lhs = (torch.randn((m, k), dtype=torch.float8_e4m3fn), torch.randn((m, (k + 127) // 128), dtype=torch.float32))
rhs = (torch.randn((n, k), dtype=torch.float8_e4m3fn), torch.randn(((n + 127) // 128, (k + 127) // 128), dtype=torch.float32))
out = torch.empty((m, n), dtype=torch.bfloat16)
gemm_fp8_fp8_bf16_nt(lhs, rhs, out)
