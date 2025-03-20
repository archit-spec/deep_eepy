from deep_gemm import gemm_fp8_fp8_bf16_nt
import torch

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU setup.")

# Define matrix dimensions
m = 1024  # Number of rows in lhs and out
k = 1024  # Number of columns in lhs and rhs
n = 1024  # Number of rows in rhs and columns in out

# Create tensors directly on GPU
lhs = (
    torch.randn((m, k), dtype=torch.float8_e4m3fn, device="cuda"), 
    torch.randn((m, (k + 127) // 128), dtype=torch.float32, device="cuda")
)
rhs = (
    torch.randn((n, k), dtype=torch.float8_e4m3fn, device="cuda"), 
    torch.randn(((n + 127) // 128, (k + 127) // 128), dtype=torch.float32, device="cuda")
)
out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")

# Run the GEMM operation
gemm_fp8_fp8_bf16_nt(lhs, rhs, out)

# If you want to verify results
print(f"Output shape: {out.shape}")
print(f"Output device: {out.device}")
