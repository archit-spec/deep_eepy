import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_gemm import gemm_fp8_fp8_bf16_nt
import time
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


class SyntheticDataset(Dataset):
    """Synthetic dataset for training"""
    def __init__(self, input_dim, output_dim, size=10000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random data on-the-fly
        x = torch.randn(self.input_dim)
        y = torch.randn(self.output_dim).to(torch.bfloat16)
        return x, y


class DeepGEMM_Linear(nn.Module):
    """Custom linear layer using DeepGEMM's FP8 GEMM operation"""
    def __init__(self, in_features, out_features):
        super().__init__()
        # Ensure dimensions are properly aligned
        self.in_features_padded = ((in_features + 127) // 128) * 128  # Ensure multiple of 128
        self.out_features_padded = ((out_features + 63) // 64) * 64   # Ensure multiple of 64
        
        # Initialize weights with scaled initialization
        self.weight = nn.Parameter(torch.randn(
            self.out_features_padded, self.in_features_padded, 
            dtype=torch.float32) * (1.0 / np.sqrt(in_features)))
        
        self.weight_scale = nn.Parameter(torch.ones(
            (self.out_features_padded + 127) // 128, 
            (self.in_features_padded + 127) // 128, 
            dtype=torch.float32))
        
        self.bias = nn.Parameter(torch.zeros(self.out_features_padded, dtype=torch.bfloat16))
        
        # Store original dimensions
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Pad input if needed
        if x.shape[1] != self.in_features_padded:
            padded_x = torch.zeros(batch_size, self.in_features_padded, 
                                  dtype=x.dtype, device=x.device)
            padded_x[:, :self.in_features] = x
            x = padded_x
            
        # Handle input
        if x.dtype != torch.float8_e4m3fn:
            x_data = x.to(torch.float8_e4m3fn)
        else:
            x_data = x
            
        # Create scale for input
        x_scale = torch.ones((batch_size, (self.in_features_padded + 127) // 128), 
                            dtype=torch.float32, device=x.device)
        
        # Convert weight to FP8 for the operation
        weight_fp8 = self.weight.to(torch.float8_e4m3fn)
        
        # Prepare inputs for DeepGEMM
        lhs = (x_data, x_scale)
        rhs = (weight_fp8, self.weight_scale)
        
        # Prepare output tensor
        out = torch.empty((batch_size, self.out_features_padded), 
                         dtype=torch.bfloat16, device=x.device)
        
        # Perform the FP8 matrix multiplication
        gemm_fp8_fp8_bf16_nt(lhs, rhs, out)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        # Return unpadded output
        return out[:, :self.out_features]


class Expert(nn.Module):
    """A single expert in the Mixture of Experts model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = DeepGEMM_Linear(input_dim, hidden_dim)
        self.fc2 = DeepGEMM_Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class MoE(nn.Module):
    """Mixture of Experts model using DeepGEMM"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=8, k=2):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = k  # Top-k experts to use
        
        # Router (gating network)
        self.router = nn.Linear(input_dim, num_experts)  # Using standard linear for router
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) 
            for _ in range(num_experts)
        ])
        
        # Load balancing loss coefficient
        self.balance_coef = 0.01
        
    def forward(self, x, return_aux_loss=True):
        batch_size = x.shape[0]
        
        # Get router scores - using standard linear for better numerical stability in routing
        router_logits = self.router(x)  # [batch_size, num_experts]
        
        # Apply softmax to get routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        vals, indices = torch.topk(router_probs, self.k, dim=-1)  # Both [batch_size, k]
        
        # Normalize the top-k probabilities
        vals = vals / vals.sum(dim=-1, keepdim=True)
        
        # Initialize expert outputs
        final_output = torch.zeros(batch_size, self.experts[0].fc2.out_features, 
                                  dtype=torch.bfloat16, device=x.device)
        
        # Compute load balancing loss
        if return_aux_loss:
            # Mean probability of routing to each expert
            router_prob_mean = router_probs.mean(dim=0)
            # Ideal uniform distribution
            uniform = torch.ones_like(router_prob_mean) / self.num_experts
            # KL divergence from uniform
            aux_loss = self.balance_coef * F.kl_div(
                router_prob_mean.log(), uniform, reduction='batchmean')
        else:
            aux_loss = 0.0
            
        # Create a more efficient batch processing approach
        # Process all experts in one go when possible
        expert_mask = torch.zeros(batch_size, self.num_experts, device=x.device)
        for b in range(batch_size):
            for i in range(self.k):
                expert_idx = indices[b, i].item()
                expert_mask[b, expert_idx] = vals[b, i]
        
        # Process batch for each expert more efficiently
        for expert_idx in range(self.num_experts):
            # Find examples that use this expert
            examples_for_expert = expert_mask[:, expert_idx].nonzero().squeeze(-1)
            if examples_for_expert.numel() > 0:
                # Get corresponding weights
                expert_weights = expert_mask[examples_for_expert, expert_idx].unsqueeze(-1)
                # Process this batch through the expert
                expert_inputs = x[examples_for_expert]
                expert_outputs = self.experts[expert_idx](expert_inputs)
                # Update the output with weighted expert outputs
                final_output[examples_for_expert] += expert_outputs * expert_weights
        
        if return_aux_loss:
            return final_output, aux_loss
        else:
            return final_output


def train(gpu, args):
    # Set up distributed training
    rank = args.nr * args.gpus + gpu
    if args.world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=rank
        )
    
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    
    # Define dimensions (must satisfy n % 64 == 0 and k % 128 == 0)
    input_dim = args.input_dim    # Must be multiple of 128
    hidden_dim = args.hidden_dim  # Must be multiple of both 64 and 128
    output_dim = args.output_dim  # Must be multiple of 64
    
    # Verify dimensions
    assert input_dim % 128 == 0, f"input_dim must be multiple of 128, got {input_dim}"
    assert hidden_dim % 128 == 0, f"hidden_dim must be multiple of 128, got {hidden_dim}"
    assert output_dim % 64 == 0, f"output_dim must be multiple of 64, got {output_dim}"
    
    # Create model
    model = MoE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=args.num_experts,
        k=args.k
    ).cuda(gpu)
    
    # Wrap model with DDP
    if args.world_size > 1:
        model = DDP(model, device_ids=[gpu])
    
    # Create optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs
    )
    
    # Create dataset and dataloader
    dataset = SyntheticDataset(
        input_dim=input_dim,
        output_dim=output_dim,
        size=args.dataset_size
    )
    
    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=rank
        )
    else:
        train_sampler = None
        
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    total_start_time = time.time()
    
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0.0
        total_aux_loss = 0.0
        
        # Use a progress tracker
        processed_batches = 0
        total_batches = len(train_loader)
        batch_times = []
        
        for inputs, targets in train_loader:
            batch_start = time.time()
            
            # Move data to GPU
            inputs = inputs.cuda(gpu, non_blocking=True)
            targets = targets.cuda(gpu, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, aux_loss = model(inputs)
            
            # Compute primary loss (MSE)
            primary_loss = F.mse_loss(outputs, targets)
            loss = primary_loss + aux_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            
            # Update stats
            total_loss += primary_loss.item()
            total_aux_loss += aux_loss.item()
            
            # Timing
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            # Progress update
            processed_batches += 1
            if gpu == 0 and processed_batches % args.log_interval == 0:
                avg_loss = total_loss / processed_batches
                avg_aux_loss = total_aux_loss / processed_batches
                avg_batch_time = sum(batch_times[-args.log_interval:]) / len(batch_times[-args.log_interval:])
                progress = 100 * processed_batches / total_batches
                
                print(f"Epoch {epoch+1}/{args.epochs} [{processed_batches}/{total_batches} ({progress:.0f}%)] | "
                      f"Loss: {avg_loss:.6f} | Aux Loss: {avg_aux_loss:.6f} | "
                      f"Time: {avg_batch_time:.4f}s/batch | "
                      f"~ETA: {avg_batch_time * (total_batches - processed_batches) / 60:.1f}min")
        
        # Update learning rate
        scheduler.step()
        
        # Epoch stats
        if gpu == 0:
            avg_loss = total_loss / len(train_loader)
            avg_aux_loss = total_aux_loss / len(train_loader)
            avg_batch_time = sum(batch_times) / len(batch_times)
            samples_per_sec = args.batch_size / avg_batch_time
            if args.world_size > 1:
                samples_per_sec *= args.world_size
                
            print(f"\nEpoch {epoch+1} completed | Avg Loss: {avg_loss:.6f} | "
                  f"Avg Aux Loss: {avg_aux_loss:.6f} | "
                  f"Throughput: {samples_per_sec:.1f} samples/sec | "
                  f"LR: {scheduler.get_last_lr()[0]:.8f}\n")
    
    # Final timing
    if gpu == 0:
        total_time = time.time() - total_start_time
        print(f"Training completed in {total_time/60:.2f} minutes")
        
        # Run inference benchmark
        model.eval()
        with torch.no_grad():
            input_size = args.batch_size
            x = torch.randn(input_size, input_dim, device=f'cuda:{gpu}')
            
            # Warmup
            for _ in range(10):
                model(x, return_aux_loss=False)
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            num_inferences = 100
            
            for _ in range(num_inferences):
                model(x, return_aux_loss=False)
                
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            
            print(f"\nInference benchmark:")
            print(f"  Time per batch: {total_time/num_inferences*1000:.2f} ms")
            print(f"  Throughput: {input_size * num_inferences / total_time:.1f} samples/sec")
            print(f"  Batch size: {input_size}")


def setup_distributed():
    """Set up distributed environment variables if needed"""
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='High Performance MOE Training')
    
    # Model configuration
    parser.add_argument('--input-dim', type=int, default=2048, 
                        help='Input dimension (must be multiple of 128)')
    parser.add_argument('--hidden-dim', type=int, default=8192, 
                        help='Hidden dimension (must be multiple of 128)')
    parser.add_argument('--output-dim', type=int, default=2048, 
                        help='Output dimension (must be multiple of 64)')
    parser.add_argument('--num-experts', type=int, default=16, 
                        help='Number of experts')
    parser.add_argument('--k', type=int, default=2, 
                        help='Number of experts to route to (top-k)')
                        
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='Batch size per GPU')
    parser.add_argument('--dataset-size', type=int, default=50000, 
                        help='Size of synthetic dataset')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, 
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, 
                        help='Weight decay')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, 
                        help='Maximum gradient norm for clipping')
                        
    # Distributed training
    parser.add_argument('--nodes', type=int, default=1, 
                        help='Number of nodes')
    parser.add_argument('--gpus', type=int, default=2, 
                        help='Number of GPUs per node')
    parser.add_argument('--nr', type=int, default=0, 
                        help='Ranking within nodes')
                        
    # Logging
    parser.add_argument('--log-interval', type=int, default=10, 
                        help='Log interval (batches)')
    
    args = parser.parse_args()
    
    # Set up distributed environment
    setup_distributed()
    
    # Calculate world size
    args.world_size = args.nodes * args.gpus
    
    # Print training configuration
    print("\n=== Training Configuration ===")
    print(f"Model dimensions: {args.input_dim}→{args.hidden_dim}→{args.output_dim}")
    print(f"Number of experts: {args.num_experts} (using top-{args.k} routing)")
    print(f"Batch size: {args.batch_size} per GPU ({args.batch_size * args.world_size} total)")
    print(f"GPUs: {args.world_size} ({args.nodes} nodes × {args.gpus} GPUs)")
    print(f"Training for {args.epochs} epochs on {args.dataset_size} examples")
    print("==============================\n")
    
    # Launch training processes
    if args.world_size > 1:
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    else:
        train(0, args)
