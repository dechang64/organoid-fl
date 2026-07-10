"""
Continuous Thought Machine (CTM) — PyTorch Implementation
Based on: Darlow et al. "Continuous Thought Machines" (Sakana AI, NeurIPS 2025)

Core innovations:
1. NLMs: Per-neuron private weights (einsum for efficiency)
2. Neural synchronization: Temporal correlation matrix (not snapshot)
3. Attention Q from synchronization (not from input) → emergent active vision
4. Loss: argmin(loss) + argmax(certainty) → adaptive computation

Usage:
    from ctm_module import CTM, CTMLoss
    model = CTM(d_model=768, d_internal=256, n_ticks=20, n_heads=8)
    logits_per_tick, certainties_per_tick = model(kv_features)
    loss = CTMLoss()(logits_per_tick, certainties_per_tick, targets)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class NeuronLevelModels(nn.Module):
    """
    Per-neuron private-weight depth-1 MLP.
    
    Each of D neurons has its own private weights (M → d_hidden → 1).
    Implemented with einsum for efficiency.
    
    Input:  pre_acts_history [B, D, M]
    Output: post_activations [B, D]
    """
    def __init__(self, d_internal: int, mem_len: int, d_hidden: int = 16):
        super().__init__()
        self.D = d_internal
        self.M = mem_len
        
        # Private weights per neuron: [M, d_hidden, D]
        # einsum 'bdM,Mhd->bdh': input [B,D,M] × weight [M,h,D] → [B,D,h]
        self.weights_1 = nn.Parameter(torch.randn(mem_len, d_hidden, d_internal) * 0.02)
        # bias must match output [B, D, d_hidden]
        self.bias_1 = nn.Parameter(torch.zeros(1, d_internal, d_hidden))
        
        # Output projection: [d_hidden, D]
        # einsum 'bdh,hd->bd': input [B,D,h] × weight [h,D] → [B,D]
        self.weights_2 = nn.Parameter(torch.randn(d_hidden, d_internal) * 0.02)
        self.bias_2 = nn.Parameter(torch.zeros(1, d_internal))
    
    def forward(self, pre_acts_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pre_acts_history: [B, D, M]
        Returns:
            post_acts: [B, D]
        """
        # Layer 1: [B, D, M] × [M, d_hidden, D] → [B, D, d_hidden]
        out = torch.einsum('bdM,Mhd->bdh', pre_acts_history, self.weights_1) + self.bias_1
        out = F.tanh(out)  # Non-linearity
        
        # Layer 2: [B, D, d_hidden] × [d_hidden, D] → [B, D]
        out = torch.einsum('bdh,hd->bd', out, self.weights_2) + self.bias_2
        out = F.tanh(out)
        
        return out


class Synchronization(nn.Module):
    """
    Neural synchronization matrix with learnable temporal decay.
    
    S_t[i,j] = (Z_i)^T · diag(R_ij) · Z_j / sqrt(sum(R_ij))
    
    where R_ij = [exp(-r_ij * (t-1)), ..., exp(0)]
    
    Also supports random neuron pair selection for bottleneck opening.
    """
    def __init__(self, d_internal: int, n_pairs: int, n_self: int = 0):
        super().__init__()
        self.D = d_internal
        
        # Learnable decay rates for each pair: [n_pairs]
        self.decay_rates = nn.Parameter(torch.zeros(n_pairs))
        
        # Random pair indices: [n_pairs, 2]
        # These are fixed at init time (not learned)
        pairs = torch.randint(0, d_internal, (n_pairs, 2))
        self.register_buffer('pair_indices', pairs)
        
        # Self pairs (i,i) for snapshot recovery
        if n_self > 0:
            self_indices = torch.randint(0, d_internal, (n_self,))
            self.register_buffer('self_indices', self_indices)
            self.n_self = n_self
        else:
            self.self_indices = None
            self.n_self = 0
        
        self.n_pairs = n_pairs
    
    def forward(self, post_acts_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            post_acts_history: [B, D, T] (T = current tick count)
        Returns:
            synch: [B, n_pairs + n_self] — synchronization values
        """
        B, D, T = post_acts_history.shape
        device = post_acts_history.device
        
        # Get pair neurons: [B, n_pairs, T] each
        i_idx = self.pair_indices[:, 0]  # [n_pairs]
        j_idx = self.pair_indices[:, 1]  # [n_pairs]
        
        z_i = post_acts_history[:, i_idx, :]  # [B, n_pairs, T]
        z_j = post_acts_history[:, j_idx, :]  # [B, n_pairs, T]
        
        # Temporal decay: R_ij[τ] = exp(-r_ij * (T-1-τ))
        # τ goes from 0 (oldest) to T-1 (newest)
        tau = torch.arange(T, device=device, dtype=torch.float32)  # [T]
        # decay: [n_pairs, T] — broadcast decay_rates with tau
        decay = torch.exp(-self.decay_rates.unsqueeze(1) * (T - 1 - tau.unsqueeze(0)))  # [n_pairs, T]
        
        # Synchronization: St_ij = sum(R * z_i * z_j) / sqrt(sum(R))
        # Paper Eq 13-15: αt_ij = sum(e^{-rij(t-τ)} * z_τ_i * z_τ_j), βt_ij = sum(e^{-rij(t-τ)})
        # St_ij = αt_ij / sqrt(βt_ij)
        weighted_z_i = z_i * decay.unsqueeze(0)  # [B, n_pairs, T]
        synch_pairs = (weighted_z_i * z_j).sum(dim=-1)  # [B, n_pairs]
        norm = torch.sqrt(decay.sum(dim=-1, keepdim=True).unsqueeze(0) + 1e-8)  # [1, n_pairs, 1]
        synch_pairs = synch_pairs / norm.squeeze(-1)
        
        # Self pairs (snapshot recovery)
        if self.n_self > 0:
            z_self = post_acts_history[:, self.self_indices, :]  # [B, n_self, T]
            # Self synchronization = latest post-activation
            synch_self = z_self[:, :, -1]  # [B, n_self]
            return torch.cat([synch_pairs, synch_self], dim=-1)  # [B, n_pairs + n_self]
        
        return synch_pairs  # [B, n_pairs]


class SynapseModel(nn.Module):
    """
    Synapse model: f_syn(concat(attn_out, z_t)) → pre_acts
    
    Can be a simple MLP or a deeper U-Net-like structure.
    """
    def __init__(self, d_internal: int, d_attn_out: int, d_hidden: int = None):
        super().__init__()
        d_in = d_internal + d_attn_out
        if d_hidden is None:
            d_hidden = d_in * 2
        
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_internal),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CTM(nn.Module):
    """
    Continuous Thought Machine.
    
    Args:
        d_model: Feature dimension from backbone (e.g., 768 for DINOv2 ViT-B/14)
        d_internal: Internal state dimension D (e.g., 256)
        n_ticks: Number of internal thought steps T (e.g., 20)
        n_heads: Attention heads (e.g., 8)
        mem_len: Pre-activation history length M (e.g., 20)
        n_action_pairs: Neuron pairs for attention Q
        n_output_pairs: Neuron pairs for output
        n_classes: Number of output classes (2 for TP/FP)
        n_self: Self pairs for snapshot recovery
        d_hidden_nlm: Hidden width of each NLM
    """
    def __init__(
        self,
        d_model: int = 768,
        d_internal: int = 256,
        n_ticks: int = 20,
        n_heads: int = 8,
        mem_len: int = 20,
        n_action_pairs: int = 128,
        n_output_pairs: int = 128,
        n_classes: int = 2,
        n_self: int = 32,
        d_hidden_nlm: int = 16,
    ):
        super().__init__()
        self.D = d_internal
        self.T = n_ticks
        self.M = mem_len
        
        # Learnable initial state
        self.z_init = nn.Parameter(torch.randn(d_internal) * 0.02)
        
        # Learnable initial pre-activation history
        self.pre_acts_init = nn.Parameter(torch.randn(d_internal, mem_len) * 0.02)
        
        # Synapse model
        self.synapses = SynapseModel(d_internal, d_model, d_hidden=d_internal * 2)
        
        # NLMs
        self.nlms = NeuronLevelModels(d_internal, mem_len, d_hidden=d_hidden_nlm)
        
        # Synchronization (action = for attention Q, output = for predictions)
        self.synch_action = Synchronization(d_internal, n_action_pairs, n_self=n_self)
        self.synch_output = Synchronization(d_internal, n_output_pairs, n_self=n_self)
        
        # Q projector: synchronization → attention query
        q_dim = n_action_pairs + n_self
        self.q_projector = nn.Linear(q_dim, d_model)
        
        # KV projector: backbone features → KV (if needed, usually identity)
        self.kv_projector = nn.Linear(d_model, d_model)
        
        # Attention module
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        
        # Output projector: synchronization → class logits
        out_dim = n_output_pairs + n_self
        self.output_proj = nn.Linear(out_dim, n_classes)
    
    def forward(self, kv_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            kv_features: [B, N, d_model] — spatial feature tokens from backbone
        
        Returns:
            logits_history: [B, n_classes, T] — logits at each tick
            certainties_history: [B, T] — certainty at each tick
            attn_weights_history: [B, T, n_heads, N] — attention weights per tick per head
        """
        B = kv_features.shape[0]
        device = kv_features.device
        
        # Project KV
        kv = self.kv_projector(kv_features)  # [B, N, d_model]
        
        # Initialize state
        z = self.z_init.unsqueeze(0).expand(B, -1)  # [B, D]
        
        # Initialize pre-activation history
        pre_acts_history = self.pre_acts_init.unsqueeze(0).expand(B, -1, -1)  # [B, D, M]
        
        # Post-activation history (grows over ticks)
        post_acts_history = torch.zeros(B, self.D, self.T, device=device)
        post_acts_history[:, :, 0] = z  # First tick uses z_init
        
        # Storage
        logits_history = []
        certainties_history = []
        attn_weights_history = []
        
        for t in range(self.T):
            # 1. Compute action synchronization from post-acts history (up to tick t+1)
            synch_a = self.synch_action(post_acts_history[:, :, :t+1])  # [B, q_dim]
            
            # 2. Project to attention Q
            q = self.q_projector(synch_a)  # [B, d_model]
            q = q.unsqueeze(1)  # [B, 1, d_model] — single query token
            
            # 3. Cross-attention: Q from internal, KV from image
            attn_out, attn_weights = self.attention(q, kv, kv, need_weights=True, average_attn_weights=False)  # [B, 1, d_model], [B, n_heads, 1, S]
            attn_out = attn_out.squeeze(1)  # [B, d_model]
            attn_weights_history.append(attn_weights.squeeze(2))  # [B, n_heads, S]
            
            # 4. Synapse: fuse attention output + current state
            synapse_input = torch.cat([attn_out, z], dim=-1)  # [B, d_model + D]
            pre_acts = self.synapses(synapse_input)  # [B, D]
            
            # 5. Update pre-activation history (FIFO: shift left, add new)
            pre_acts_history = torch.cat([
                pre_acts_history[:, :, 1:],  # Drop oldest
                pre_acts.unsqueeze(-1)  # Add newest
            ], dim=-1)  # [B, D, M]
            
            # 6. NLMs: compute post-activations
            z = self.nlms(pre_acts_history)  # [B, D]
            
            # 7. Store post-activation
            post_acts_history[:, :, t] = z
            
            # 8. Compute output synchronization
            synch_o = self.synch_output(post_acts_history[:, :, :t+1])  # [B, out_dim]
            
            # 9. Project to class logits
            logits = self.output_proj(synch_o)  # [B, n_classes]
            
            # 10. Compute certainty (1 - normalized entropy)
            p = F.softmax(logits, dim=-1)
            log_p = torch.log_softmax(logits, dim=-1)
            entropy = -(p * log_p).sum(dim=-1)  # [B]
            max_entropy = math.log(logits.shape[-1])
            certainty = 1 - (entropy / max_entropy)  # [B]
            
            logits_history.append(logits)
            certainties_history.append(certainty)
        
        # Stack: [B, T, n_classes] → [B, n_classes, T]
        logits_history = torch.stack(logits_history, dim=1).permute(0, 2, 1)  # [B, C, T]
        certainties_history = torch.stack(certainties_history, dim=1)  # [B, T]
        attn_weights_history = torch.stack(attn_weights_history, dim=1)  # [B, T, n_heads, S]
        
        return logits_history, certainties_history, attn_weights_history


class CTMLoss(nn.Module):
    """
    CTM loss function: select tick with min loss + tick with max certainty.
    
    L = (L[t1] + L[t2]) / 2
    
    where t1 = argmin(L), t2 = argmax(C)
    
    This enables native adaptive computation — different samples can use
    different numbers of ticks.
    """
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    def forward(
        self,
        logits_history: torch.Tensor,  # [B, C, T]
        certainties_history: torch.Tensor,  # [B, T]
        targets: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, dict]:
        B, C, T = logits_history.shape
        
        # Expand targets over ticks
        targets_exp = targets.unsqueeze(-1).expand(-1, T)  # [B, T]
        
        # Compute loss at each tick
        # Need to reshape: [B, C, T] → [B*T, C] for CrossEntropyLoss
        logits_flat = logits_history.permute(0, 2, 1).reshape(B * T, C)  # [B*T, C]
        targets_flat = targets_exp.reshape(B * T)  # [B*T]
        losses_flat = self.loss_fn(logits_flat, targets_flat)  # [B*T]
        losses = losses_flat.reshape(B, T)  # [B, T]
        
        # t1 = argmin(loss) per sample
        t1 = losses.argmin(dim=-1)  # [B]
        loss_t1 = losses[torch.arange(B), t1]  # [B]
        
        # t2 = argmax(certainty) per sample
        t2 = certainties_history.argmax(dim=-1)  # [B]
        loss_t2 = losses[torch.arange(B), t2]  # [B]
        
        # Final loss: average
        loss = (loss_t1 + loss_t2).mean() / 2
        
        # Metrics for logging
        with torch.no_grad():
            acc_t1 = (logits_history[torch.arange(B), :, t1].argmax(-1) == targets).float().mean()
            acc_t2 = (logits_history[torch.arange(B), :, t2].argmax(-1) == targets).float().mean()
            acc_final = (logits_history[:, :, -1].argmax(-1) == targets).float().mean()
        
        info = {
            'loss': loss.item(),
            'acc_best': acc_t1.item(),
            'acc_certain': acc_t2.item(),
            'acc_final': acc_final.item(),
            'mean_tick_t1': t1.float().mean().item(),
            'mean_tick_t2': t2.float().mean().item(),
            'mean_certainty': certainties_history.mean().item(),
        }
        
        return loss, info


# === Test ===
if __name__ == '__main__':
    print("=" * 60)
    print("CTM Module Test")
    print("=" * 60)
    
    # Dummy data
    B, N, d_model = 4, 50, 768  # batch=4, 50 tokens, 768-dim features
    kv_features = torch.randn(B, N, d_model)
    targets = torch.randint(0, 2, (B,))
    
    # Create model
    model = CTM(
        d_model=768,
        d_internal=256,
        n_ticks=20,
        n_heads=8,
        mem_len=20,
        n_action_pairs=128,
        n_output_pairs=128,
        n_classes=2,
        n_self=32,
        d_hidden_nlm=16,
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Forward pass
    print(f"\nInput: kv_features {kv_features.shape}, targets {targets.shape}")
    logits_hist, cert_hist, attn_hist = model(kv_features)
    print(f"Output: logits_history {logits_hist.shape}, certainties_history {cert_hist.shape}, attn {attn_hist.shape}")
    
    # Loss
    loss_fn = CTMLoss(n_classes=2)
    loss, info = loss_fn(logits_hist, cert_hist, targets)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Info: {info}")
    
    # Backward
    loss.backward()
    print(f"\nBackward pass: OK")
    
    # Check gradients
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"Gradient norm: {grad_norm:.4f}")
    
    # Print tick-wise accuracy
    print(f"\nTick-wise predictions (sample 0):")
    for t in range(model.T):
        logits_t = logits_hist[0, :, t]
        pred_t = logits_t.argmax().item()
        prob_t = F.softmax(logits_t, dim=-1)
        cert_t = cert_hist[0, t].item()
        print(f"  Tick {t:2d}: pred={pred_t}, prob={prob_t.max().item():.3f}, certainty={cert_t:.3f}")
    
    print("\n✓ CTM module test passed!")
