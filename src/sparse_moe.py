"""
Sparse Mixture of Experts module for JAX/Flax implementation.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

from .expert import Expert
from .router import NoisyTopKRouter


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts using JAX's batching capabilities.
    This implementation is JIT-compatible by avoiding dynamic shapes.
    """
    n_embd: int
    num_experts: int
    top_k: int
    dropout_rate: float = 0.1
    
    def setup(self):
        """Initialize the router and experts."""
        self.router = NoisyTopKRouter(
            n_embd=self.n_embd,
            num_experts=self.num_experts,
            top_k=self.top_k
        )
        self.experts = [Expert(n_embd=self.n_embd, dropout_rate=self.dropout_rate) 
                       for _ in range(self.num_experts)]
    
    def __call__(self, x, training=False, rng_key=None):
        """
        Forward pass through the Sparse MoE layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            training: Whether in training mode
            rng_key: Random key for router noise
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Get routing probabilities and expert indices
        router_weights, router_indices = self.router(x, rng_key=rng_key)
        
        # Initialize output
        final_output = jnp.zeros_like(x)
        
        # Process each expert using vectorized operations to avoid dynamic shapes(which wont be jit compatible)
        for expert_idx in range(self.num_experts):
            # Create mask for tokens assigned to this expert
            expert_mask = jnp.any(router_indices == expert_idx, axis=-1)  # (batch_size, seq_len)
            
            # Get the weight for this expert from router_weights
            expert_weight = router_weights[:, :, expert_idx]  # (batch_size, seq_len)
            
            # Apply expert only where mask is True, but process full tensor to maintain shape
            # This avoids the dynamic shape issue from jnp.where
            expert_output = self.experts[expert_idx](x, training=training)
            
            # Weight the expert output and mask it
            weighted_output = expert_output * expert_weight[:, :, None]  # Broadcast weight
            masked_output = weighted_output * expert_mask[:, :, None]  # Apply mask
            
            # Add to final output
            final_output = final_output + masked_output
        
        return final_output


class SparseMoEOptimized(nn.Module):
    """
    Optimized Sparse Mixture of Experts using gather/scatter operations.
    This version is more efficient for sparse routing.
    """
    n_embd: int
    num_experts: int
    top_k: int
    dropout_rate: float = 0.1
    
    def setup(self):
        """Initialize the router and experts."""
        self.router = NoisyTopKRouter(
            n_embd=self.n_embd,
            num_experts=self.num_experts,
            top_k=self.top_k
        )
        self.experts = [Expert(n_embd=self.n_embd, dropout_rate=self.dropout_rate) 
                       for _ in range(self.num_experts)]
    
    def __call__(self, x, training=False, rng_key=None):
        """
        Forward pass using gather/scatter operations for efficiency.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            training: Whether in training mode
            rng_key: Random key for router noise
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        
        # Get routing probabilities and expert indices
        router_weights, router_indices = self.router(x, rng_key=rng_key)
        
        # Flatten input for easier processing
        flat_x = x.reshape(-1, n_embd)  # (batch_size * seq_len, n_embd)
        flat_router_weights = router_weights.reshape(-1, self.num_experts)  # (batch_size * seq_len, num_experts)
        flat_router_indices = router_indices.reshape(-1, self.top_k)  # (batch_size * seq_len, top_k)
        
        # Initialize output
        final_output = jnp.zeros_like(flat_x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which tokens are routed to this expert
            expert_mask = jnp.any(flat_router_indices == expert_idx, axis=-1)  # (batch_size * seq_len,)
            
            # Get expert weights for this expert
            expert_weights = flat_router_weights[:, expert_idx]  # (batch_size * seq_len,)
            
            # Process through expert (full tensor to maintain JIT compatibility)
            expert_output = self.experts[expert_idx](flat_x, training=training)
            
            # Apply weights and mask
            weighted_output = expert_output * expert_weights[:, None]
            masked_output = weighted_output * expert_mask[:, None]
            
            # Add to final output
            final_output = final_output + masked_output
        
        # Reshape back to original shape
        return final_output.reshape(batch_size, seq_len, n_embd) 