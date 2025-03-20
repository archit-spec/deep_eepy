import torch
from deep_gemm import m_grouped_gemm_fp8_fp8_bf16_nt_masked  # Adjust import based on your setup

# Define variables with example values
num_groups = 4   # Number of groups
m_max = 128      # Maximum size for m dimension
k = 256          # Size of k dimension
n = 128          # Size of n dimension

# Create tensors
lhs = (torch.randn((num_groups, m_max, k), dtype=torch.float8_e4m3fn),
       torch.randn((num_groups, m_max, (k + 127) // 128), dtype=torch.float32))
rhs = (torch.randn((num_groups, n, k), dtype=torch.float8_e4m3fn),
       torch.randn((num_groups, (n + 127) // 128, (k + 127) // 128), dtype=torch.float32))
out = torch.empty((num_groups, m_max, n), dtype=torch.bfloat16)
masked_m = torch.randint(0, m_max, (num_groups,), dtype=torch.int32)
expected_m = m_max // 2  # Example expected value

# Call the function
m_grouped_gemm_fp8_fp8_bf16_nt_masked(lhs, rhs, out, masked_m, expected_m)
