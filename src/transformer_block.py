"""
Transformer block combining attention and Sparse MoE layers.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .attention import MultiHeadAttention
from .sparse_moe import SparseMoE


class Block(nn.Module):
    """
    Mixture of Experts Transformer block: communication followed by computation.
    (multi-head self attention + SparseMoE)
    """
    n_embd: int
    n_head: int
    num_experts: int
    block_size: int
    top_k: int
    dropout_rate: float = 0.1

    def setup(self):
        """Initialize the attention and MoE layers."""
        head_size = self.n_embd // self.n_head
        
        self.sa = MultiHeadAttention(
            n_embd=self.n_embd,
            n_head=self.n_head,
            head_size=head_size,
            block_size=self.block_size,
            dropout_rate=self.dropout_rate
        )
        
        self.smoe = SparseMoE(
            n_embd=self.n_embd,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dropout_rate=self.dropout_rate
        )

        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
    
    def __call__(self, x, rng_key, training: bool = False):
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            rng_key: Random key for MoE routing
            training: Whether in training mode
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        sa_key, moe_key = jax.random.split(rng_key)

        # Self attention with residual connection and pre-norm
        x = x + self.sa(self.ln1(x), training=training)

        # Sparse MoE with residual connection and pre-norm
        x = x + self.smoe(self.ln2(x), training=training, rng_key=moe_key)

        return x 