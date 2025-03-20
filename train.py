import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_gemm import gemm_fp8_fp8_bf16_nt
import time
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
import os
from transformers import AutoTokenizer
from datasets import load_dataset
import random

class FinewebDataset(Dataset):
    """Dataset wrapper for fineweb-1M_longish"""
    def __init__(self, dataset_path="BEEspoke/fineweb-1M_longish", 
                 split="train", max_samples=200000, max_length=2048,
                 tokenizer_name="gpt2"):
        super().__init__()
        
        print(f"Loading dataset: {dataset_path}")
        self.dataset = load_dataset(dataset_path, split=split)
        
        # Limit to max_samples
        if max_samples and max_samples < len(self.dataset):
            print(f"Limiting dataset to {max_samples} samples (from {len(self.dataset)})")
            indices = random.sample(range(len(self.dataset)), max_samples)
            self.dataset = self.dataset.select(indices)
        
        print(f"Loaded {len(self.dataset)} samples")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]
        
        # Tokenize text
        tokens = self.tokenizer(
            text, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].squeeze()
        attention_mask = tokens["attention_mask"].squeeze()
        
        # Create inputs (all tokens except last) and targets (all tokens except first)
        inputs = input_ids[:-1].clone()
        targets = input_ids[1:].clone()
        mask = attention_mask[1:].clone()
        
        return {
            "input_ids": inputs,
            "targets": targets,
            "attention_mask": mask
        }


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
        
        # Router (gating network) - use standard PyTorch Linear layer for routing
        self.router = nn.Linear(input_dim, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) 
            for _ in range(num_experts)
        ])
        
        # Load balancing loss coefficient
        self.balance_coef = 0.01
        
    def forward(self, x, return_aux_loss=True):
        # Get original shape
        original_shape = x.shape
        batch_size, seq_len, d_model = original_shape
        
        # Reshape for routing
        x_reshaped = x.reshape(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Get router scores
        router_logits = self.router(x_reshaped)  # [batch_size * seq_len, num_experts]
        
        # Apply softmax to get routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        vals, indices = torch.topk(router_probs, self.k, dim=-1)  # Both [batch_size * seq_len, k]
        
        # Normalize the top-k probabilities
        vals = vals / vals.sum(dim=-1, keepdim=True)
        
        # Initialize expert outputs with correct dtype
        final_output = torch.zeros_like(x_reshaped, dtype=torch.bfloat16)
        
        # Compute load balancing loss
        if return_aux_loss:
            # Mean probability of routing to each expert
            router_prob_mean = router_probs.mean(dim=0)
            # Ideal uniform distribution
            uniform = torch.ones_like(router_prob_mean) / self.num_experts
            # KL divergence from uniform
            aux_loss = self.balance_coef * F.kl_div(
                router_prob_mean.log(), uniform, reduction='batchmean'
            )
        else:
            aux_loss = torch.tensor(0.0, device=x.device)
                
        # Dispatch to experts
        # Use a flattened approach that processes all examples for each expert in batches
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is selected in top-k
            for k_idx in range(self.k):
                # Get the indices where the current expert is selected at position k_idx
                expert_mask = (indices[:, k_idx] == expert_idx)
                if not expert_mask.any():
                    continue
                    
                # Get corresponding inputs
                expert_inputs = x_reshaped[expert_mask]
                
                # Get corresponding weights
                expert_weights = vals[expert_mask, k_idx].unsqueeze(-1)
                
                # Process through the expert
                expert_output = self.experts[expert_idx](expert_inputs)
                
                # Add weighted outputs
                final_output[expert_mask] += expert_output * expert_weights
                
        # Reshape back to original dimensions
        final_output = final_output.reshape(original_shape)
                
        if return_aux_loss:
            return final_output, aux_loss
        else:
            return final_output


class TransformerMoEModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_experts, top_k, n_layers, max_seq_len):
        super().__init__()
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embedding
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Transformer layers with MoE FFN
        self.layers = nn.ModuleList([
            TransformerLayerWithMoE(d_model, n_heads, d_ff, num_experts, top_k)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.out_proj = nn.Linear(d_model, vocab_size)  # Standard linear for output projection
        
        # Initialize parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, DeepGEMM_Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        x = self.token_embedding(input_ids)
        
        # Add position embeddings (truncate if sequence is shorter)
        position_emb = self.position_embedding[:, :seq_len, :]
        x = x + position_emb
        
        # Apply transformer layers
        moe_losses = []
        for layer in self.layers:
            x, moe_loss = layer(x, attention_mask)
            moe_losses.append(moe_loss)
            
        # Project to vocabulary
        logits = self.out_proj(x)
        
        # Compute total MoE loss
        total_moe_loss = torch.stack(moe_losses).sum()
        
        return logits, total_moe_loss


class TransformerLayerWithMoE(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_experts, top_k):
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # MoE FFN
        self.moe = MoE(
            input_dim=d_model,
            hidden_dim=d_ff,
            output_dim=d_model,
            num_experts=num_experts,
            k=top_k
        )
        
    def forward(self, x, attention_mask=None):
        # Self attention with residual connection
        residual = x
        x = self.norm1(x)
        
        # Process attention mask for PyTorch's MultiheadAttention
        key_padding_mask = None
        if attention_mask is not None:
            # PyTorch expects mask values of True to be masked positions
            key_padding_mask = ~attention_mask.bool()
            
        # Apply self attention
        x, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + residual
        
        # MoE FFN with residual connection
        residual = x
        x = self.norm2(x)
        x, moe_loss = self.moe(x)
        x = x + residual
        
        return x, moe_loss


def train_model_on_fineweb(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Load dataset
    dataset = FinewebDataset(
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        max_length=args.max_length,
        tokenizer_name=args.tokenizer
    )
    
    # Get tokenizer vocabulary size
    vocab_size = len(dataset.tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create data loader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = TransformerMoEModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        num_experts=args.num_experts,
        top_k=args.top_k,
        n_layers=args.n_layers,
        max_seq_len=args.max_length
    ).to(device)
    
    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Training loop
    total_start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_moe_loss = 0.0
        
        # Progress tracking
        processed_batches = 0
        total_batches = len(data_loader)
        batch_times = []
        
        for batch in data_loader:
            batch_start = time.time()
            
            # Get batch data
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                # Model outputs
                logits, moe_loss = model(input_ids, attention_mask)
                
                # Reshape for cross entropy
                logits = logits.view(-1, vocab_size)
                targets = targets.reshape(-1)
                
                # Compute loss
                ce_loss = F.cross_entropy(logits, targets, ignore_index=dataset.tokenizer.pad_token_id)
                loss = ce_loss + args.moe_loss_weight * moe_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                
                # Update stats
                total_loss += ce_loss.item()
                total_moe_loss += moe_loss.item()
            except Exception as e:
                print(f"Error during training: {e}")
                print(f"Batch shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
                print(f"Will skip this batch and continue")
                continue
            
            # Timing
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            # Progress update
            processed_batches += 1
            if processed_batches % args.log_interval == 0:
                avg_loss = total_loss / processed_batches
                avg_moe_loss = total_moe_loss / processed_batches
                avg_batch_time = sum(batch_times[-args.log_interval:]) / len(batch_times[-args.log_interval:])
                progress = 100 * processed_batches / total_batches
                
                print(f"Epoch {epoch+1}/{args.epochs} [{processed_batches}/{total_batches} ({progress:.0f}%)] | "
                      f"CE Loss: {avg_loss:.6f} | MoE Loss: {avg_moe_loss:.6f} | "
                      f"Time: {avg_batch_time:.4f}s/batch | "
                      f"~ETA: {avg_batch_time * (total_batches - processed_batches) / 60:.1f}min")
        
        # Update learning rate
        scheduler.step()
        
        # Epoch stats
        avg_loss = total_loss / max(processed_batches, 1)  # Avoid division by zero
        avg_moe_loss = total_moe_loss / max(processed_batches, 1)
        avg_batch_time = sum(batch_times) / max(len(batch_times), 1)
        samples_per_sec = args.batch_size / max(avg_batch_time, 1e-10)  # Avoid division by zero
        
        print(f"\nEpoch {epoch+1} completed | Avg CE Loss: {avg_loss:.6f} | "
              f"Avg MoE Loss: {avg_moe_loss:.6f} | "
              f"Throughput: {samples_per_sec:.1f} samples/sec | "
              f"LR: {scheduler.get_last_lr()[0]:.8f}\n")
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        os.makedirs(args.output_dir, exist_ok=True)
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    # Final timing
    total_time = time.time() - total_start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MoE model on fineweb-1M_longish')
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, default="BEE-spoke-data/fineweb-1M_longish",
                        help='Path to the dataset on HuggingFace or local path')
    parser.add_argument('--max_samples', type=int, default=200000,
                        help='Maximum number of samples to use from the dataset')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--tokenizer', type=str, default="gpt2",
                        help='Tokenizer to use')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=1024,
                        help='Model dimension (must be multiple of 128)')
    parser.add_argument('--n_heads', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='Feed-forward dimension (must be multiple of 128)')
    parser.add_argument('--num_experts', type=int, default=16,
                        help='Number of MoE experts')
    parser.add_argument('--top_k', type=int, default=2,
                        help='Number of experts to route to (top-k)')
    parser.add_argument('--n_layers', type=int, default=12,
                        help='Number of transformer layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--moe_loss_weight', type=float, default=0.01,
                        help='Weight for MoE auxiliary loss')
    
    # Miscellaneous
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval (batches)')
    parser.add_argument('--output_dir', type=str, default="./fineweb_moe_output",
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    # Ensure dimensions are properly aligned
    assert args.d_model % 128 == 0, f"d_model must be multiple of 128, got {args.d_model}"
    assert args.d_ff % 128 == 0, f"d_ff must be multiple of 128, got {args.d_ff}"
    
    # Print training configuration
    print("\n=== Training Configuration ===")
    print(f"Dataset: {args.dataset_path} (using {args.max_samples} samples)")
    print(f"Model: Transformer with MoE")
    print(f"  - Dimensions: d_model={args.d_model}, d_ff={args.d_ff}")
    print(f"  - Layers: {args.n_layers} with {args.n_heads} heads")
    print(f"  - MoE: {args.num_experts} experts, top-{args.top_k} routing")
    print(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Output directory: {args.output_dir}")
    print("==============================\n")
    
    # Start training
    train_model_on_fineweb(args)
