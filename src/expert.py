"""
Expert module for Sparse Mixture of Experts implementation in JAX/Flax.
"""

import jax.numpy as jnp
import flax.linen as nn


class Expert(nn.Module):
    """
    An MLP expert module - a simple linear layer followed by a non-linearity.
    Each expert is a feed-forward network that processes tokens routed to it.
    """
    n_embd: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        """
        Forward pass through the expert.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd) or (num_tokens, n_embd)
            training: Whether in training mode (for dropout)
            
        Returns:
            Output tensor of same shape as input
        """
        # First linear layer with 4x expansion
        x = nn.Dense(4 * self.n_embd, name="w1")(x)
        # Non-linearity
        x = nn.relu(x)
        # Second linear layer back to original dimension
        x = nn.Dense(self.n_embd, name="w2")(x)
        # Dropout
        x = nn.Dropout(
            rate=self.dropout_rate,
            deterministic=not training
        )(x)
        return x 