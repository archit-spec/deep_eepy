import torch
from deep_gemm import get_col_major_tma_aligned_tensor, per_token_cast_to_fp8, per_block_cast_to_fp8

class MoEModel:
    def __init__(self, num_experts, input_dim, hidden_dim, device='cuda'):
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.experts = [torch.nn.Linear(input_dim, hidden_dim).to(device=device, dtype=torch.bfloat16)
                        for _ in range(num_experts)]

    def forward(self, x, gate_scores):
        # x: [batch_size, input_dim], gate_scores: [batch_size, num_experts]
        batch_size = x.size(0)

        # Prepare x_fp8
        x_fp8_data, x_fp8_scale = per_token_cast_to_fp8(x)
        x_fp8_data = x_fp8_data.unsqueeze(0).expand(self.num_experts, -1, -1).contiguous().view(-1, self.input_dim)
        x_fp8_scale = x_fp8_scale.unsqueeze(0).expand(self.num_experts, -1, -1).contiguous().view(-1, x_fp8_scale.size(-1))
        x_fp8 = (x_fp8_data, get_col_major_tma_aligned_tensor(x_fp8_scale))

        # Prepare y_fp8
        y_fp8_data = []
        y_fp8_scale = []
        for expert in self.experts:
            weight = expert.weight.t()
            weight_fp8, scale = per_block_cast_to_fp8(weight)
            y_fp8_data.append(weight_fp8)
            y_fp8_scale.append(scale)
        y_fp8_data = torch.stack(y_fp8_data)
        y_fp8_scale = torch.stack(y_fp8_scale)
        y_fp8 = (y_fp8_data, y_fp8_scale)

        # Prepare out and m_indices
        out = torch.empty(self.num_experts * batch_size, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        m_indices = torch.arange(0, self.num_experts, device=self.device, dtype=torch.int)
        m_indices = m_indices.unsqueeze(-1).expand(self.num_experts, batch_size).contiguous().view(-1)

        # Perform grouped GEMM
        import deep_gemm
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

        # Post-process
        out = out.view(self.num_experts, batch_size, self.hidden_dim)
        output = (gate_scores.T.unsqueeze(-1) * out).sum(dim=0)
        return output

# Test
device = 'cuda'
batch_size, input_dim, hidden_dim, num_experts = 8192, 7168, 4096, 4
x = torch.randn(batch_size, input_dim, device=device, dtype=torch.bfloat16)
gate_scores = torch.softmax(torch.randn(batch_size, num_experts, device=device), dim=-1)
model = MoEModel(num_experts, input_dim, hidden_dim, device)
output = model.forward(x, gate_scores)
print(output.shape)  # Should be [batch_size, hidden_dim]
