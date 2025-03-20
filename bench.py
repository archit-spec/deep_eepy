from deep_gemm import gemm_fp8_fp8_bf16_nt
import torch

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU setup.")

# Define matrix dimensions
m = 1024  # Number of rows in lhs and out
k = 1024  # Number of columns in lhs and rhs
n = 1024  # Number of rows in rhs and columns in out

# Create float32 tensors and convert to fp8
lhs_data = torch.randn((m, k), dtype=torch.float32, device="cuda")
lhs_scale = torch.ones((m, (k + 127) // 128), dtype=torch.float32, device="cuda")
lhs = (lhs_data.to(torch.float8_e4m3fn), lhs_scale)

rhs_data = torch.randn((n, k), dtype=torch.float32, device="cuda")
rhs_scale = torch.ones(((n + 127) // 128, (k + 127) // 128), dtype=torch.float32, device="cuda")
rhs = (rhs_data.to(torch.float8_e4m3fn), rhs_scale)

# Create output tensor
out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

# Run the GEMM operation
gemm_fp8_fp8_bf16_nt(lhs, rhs, out)

# Print some debug info
print(f"Output shape: {out.shape}")
print(f"Output device: {out.device}")
print(f"Output mean: {out.mean().item()}")
print(f"Output min: {out.min().item()}, max: {out.max().item()}")
