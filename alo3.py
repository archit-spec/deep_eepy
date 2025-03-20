from deep_gemm import m_grouped_gemm_fp8_fp8_bf16_nt_contiguous
import torch
lhs = (torch.randn((m_sum, k), dtype=torch.float8_e4m3fn), torch.randn((m_sum, (k + 127) // 128), dtype=torch.float32))
rhs = (torch.randn((num_groups, n, k), dtype=torch.float8_e4m3fn), torch.randn((num_groups, (n + 127) // 128, (k + 127) // 128), dtype=torch.float32))
out = torch.empty((m_sum, n), dtype=torch.bfloat16)
m_indices = torch.randint(0, num_groups, (m_sum,), dtype=torch.int32)
m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(lhs, rhs, out, m_indices)
