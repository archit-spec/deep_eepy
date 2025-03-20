import torch
from deep_gemm import m_grouped_gemm_fp8_fp8_bf16_nt_masked

lhs = (torch.randn((num_groups, m_max, k), dtype=torch.float8_e4m3fn), 
       torch.randn((num_groups, m_max, (k + 127) // 128), dtype=torch.float32))
rhs = (torch.randn((num_groups, n, k), dtype=torch.float8_e4m3fn), 
       torch.randn((num_groups, (n + 127) // 128, (k + 127) // 128), dtype=torch.float32))
out = torch.empty((num_groups, m_max, n), dtype=torch.bfloat16)
masked_m = torch.randint(0, m_max, (num_groups,), dtype=torch.int32)
expected_m = m_max // 2  # Example expected value
m_grouped_gemm_fp8_fp8_bf16_nt_masked(lhs, rhs, out, masked_m, expected_m)
