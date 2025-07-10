"""
Attention modules for the Sparse MoE Language Model in JAX/Flax.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class Head(nn.Module):
    """
    One head of causal self-attention.
    """
    head_size: int
    n_embd: int
    block_size: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        Forward pass through a single attention head.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            training: Whether in training mode (for dropout)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, head_size)
        """
        B, T, C = x.shape

        # Create key, query, and value projections
        key = nn.Dense(self.head_size, use_bias=False, name="key")(x)
        query = nn.Dense(self.head_size, use_bias=False, name="query")(x)
        value = nn.Dense(self.head_size, use_bias=False, name="value")(x)

        # Compute attention scores
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = query @ jnp.swapaxes(key, -2, -1) * (self.head_size**-0.5)
        
        # Create causal mask for the current sequence length
        T_use = min(T, self.block_size)
        mask = jnp.tril(jnp.ones((T_use, T_use)))
        
        # Broadcast mask to batch dimension
        mask = jnp.broadcast_to(mask, (B, T_use, T_use))
        
        # Apply causal mask
        wei = jnp.where(mask, wei, -jnp.inf)

        # Apply softmax
        wei = jax.nn.softmax(wei, axis=-1)
        
        # Apply dropout
        wei = nn.Dropout(self.dropout_rate, deterministic=not training)(wei)

        # Apply attention to values
        out = wei @ value
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    """
    n_embd: int
    n_head: int
    head_size: int
    block_size: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=False):
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            training: Whether in training mode
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        # Create multiple heads
        heads = [Head(
            head_size=self.head_size,
            n_embd=self.n_embd,
            block_size=self.block_size,
            dropout_rate=self.dropout_rate
        ) for _ in range(self.n_head)]
        
        # Apply each head and concatenate results
        head_outputs = [head(x, training=training) for head in heads]
        out = jnp.concatenate(head_outputs, axis=-1)
        
        # Project back to embedding dimension
        out = nn.Dense(self.n_embd)(out)
        out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(out)
        
        return out


class MultiHeadAttentionOptimized(nn.Module):
    """
    Optimized multi-head attention using single QKV projection.
    More efficient than separate heads.
    """
    n_embd: int
    n_head: int
    block_size: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=False):
        """
        Forward pass through optimized multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
            training: Whether in training mode
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_embd)
        """
        B, T, C = x.shape
        head_size = self.n_embd // self.n_head
        
        # Single projection for all heads
        qkv = nn.Dense(3 * self.n_embd, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head attention
        q = q.reshape(B, T, self.n_head, head_size).transpose(0, 2, 1, 3)  # (B, n_head, T, head_size)
        k = k.reshape(B, T, self.n_head, head_size).transpose(0, 2, 1, 3)  # (B, n_head, T, head_size)
        v = v.reshape(B, T, self.n_head, head_size).transpose(0, 2, 1, 3)  # (B, n_head, T, head_size)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (head_size**-0.5)
        
        # Apply causal mask
        T_use = min(T, self.block_size)
        mask = jnp.tril(jnp.ones((T_use, T_use)))
        att = jnp.where(mask, att, -jnp.inf)
        
        # Apply softmax and dropout
        att = jax.nn.softmax(att, axis=-1)
        att = nn.Dropout(self.dropout_rate, deterministic=not training)(att)
        
        # Apply attention to values
        out = att @ v  # (B, n_head, T, head_size)
        
        # Reshape back to original format
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # Final projection
        out = nn.Dense(self.n_embd)(out)
        out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(out)
        
        return out 