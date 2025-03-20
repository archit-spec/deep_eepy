import torch
import torch.nn as nn
import deep_gemm  # Assuming this is the module containing the GEMM functions

class MoEWithDeepGEMM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(MoEWithDeepGEMM, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Move all model parameters to the same device as input
        device = x.device
        self.to(device)

        # Gating
        gate_scores = torch.softmax(self.gate(x), dim=-1)  # [batch_size, num_experts]

        # Prepare contiguous input tensor for all experts
        batch_size = x.shape[0]
        m_per_group = batch_size  # Tokens per expert
        alignment = deep_gemm.get_m_alignment_for_contiguous_layout()
        if m_per_group % alignment != 0:
            padding = alignment - (m_per_group % alignment)
            x = torch.nn.functional.pad(x, (0, 0, 0, padding))  # Pad to align
            m_per_group = x.shape[0]

        # Stack expert weights and cast to FP8
        expert_weights = torch.stack([expert.weight for expert in self.experts])  # [num_experts, hidden_dim, input_dim]
        expert_weights_fp8 = expert_weights.to(dtype=torch.float8_e4m3fn, device=device)  # Example FP8 type
        x_fp8 = x.to(dtype=torch.float8_e4m3fn, device=device)

        # DeepGEMM expects transposed RHS (weights), so we use NT format
        rhs = expert_weights_fp8  # [num_experts, hidden_dim, input_dim]
        lhs = x_fp8.unsqueeze(0).expand(self.num_experts, -1, -1)  # [num_experts, m_per_group, input_dim]

        # Scaling factors (assuming only rhs_scale is needed)
        rhs_scale = torch.ones(self.num_experts, self.hidden_dim, dtype=torch.bfloat16, device=device)

        # Call DeepGEMM grouped GEMM with correct parameters
        output = deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(lhs=lhs, rhs=rhs, rhs_scale=rhs_scale)

        # Combine with gate scores
        output = (gate_scores.T.unsqueeze(-1) * output).sum(dim=0)  # [m_per_group, hidden_dim]
        return output[:batch_size]  # Remove padding if added

# Initialize model and move to GPU
model = MoEWithDeepGEMM(input_dim=512, hidden_dim=1024, num_experts=4).to("cuda")

# Create input tensor on GPU
x = torch.randn(32, 512, device="cuda")

# Verify devices
print(next(model.parameters()).device)  # Should be 'cuda:0'
print(x.device)  # Should be 'cuda:0'

# Run forward pass
output = model(x)
print("Forward pass successful!")
