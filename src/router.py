"""
Router modules for Sparse Mixture of Experts implementation in JAX/Flax.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


class TopKRouter(nn.Module):
    """
    Router that selects top-k experts for each token.
    """
    n_embd: int
    num_experts: int
    top_k: int

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through the router.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            
        Returns:
            router_output: Sparse softmax weights of shape (batch_size, seq_len, num_experts)
            top_k_indices: Indices of top-k experts of shape (batch_size, seq_len, top_k)
        """
        # Project to expert logits
        logits = nn.Dense(self.num_experts)(x)  # (batch, seq_len, num_experts)
        
        # Get top-k logits and indices
        top_k_logits, top_k_indices = jax.lax.top_k(logits, self.top_k)
        
        # Create sparse logits by masking non-top-k values
        zeros = jnp.full_like(logits, -jnp.inf)
        sparse_logits = jnp.put_along_axis(zeros, top_k_indices, top_k_logits, axis=-1, inplace=False)
        
        # Apply softmax to get routing weights
        router_output = jax.nn.softmax(sparse_logits, axis=-1)
        
        return router_output, top_k_indices


class NoisyTopKRouter(nn.Module):
    """
    Router that selects top-k experts for each token with noise for load balancing.
    """
    n_embd: int
    num_experts: int
    top_k: int

    @nn.compact
    def __call__(self, x, rng_key):
        """
        Forward pass through the noisy router.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            rng_key: Random key for noise generation
            
        Returns:
            router_output: Sparse softmax weights of shape (batch_size, seq_len, num_experts)
            top_k_indices: Indices of top-k experts of shape (batch_size, seq_len, top_k)
        """
        # Project to expert logits
        logits = nn.Dense(self.num_experts)(x)  # (batch, seq_len, num_experts)
        
        # Project to noise logits
        noise_logits = nn.Dense(self.num_experts)(x)  # (batch, seq_len, num_experts)
        
        # Add scaled Gaussian noise to logits
        noise = jax.random.normal(rng_key, logits.shape)
        noise = noise * jax.nn.softplus(noise_logits)
        noisy_logits = logits + noise
        
        # Get top-k logits and indices
        top_k_logits, top_k_indices = jax.lax.top_k(noisy_logits, self.top_k)
        
        # Create sparse logits by masking non-top-k values
        zeros = jnp.full_like(noisy_logits, -jnp.inf)
        sparse_logits = jnp.put_along_axis(zeros, top_k_indices, top_k_logits, axis=-1, inplace=False)
        
        # Apply softmax to get routing weights
        router_output = jax.nn.softmax(sparse_logits, axis=-1)
        
        return router_output, top_k_indices 