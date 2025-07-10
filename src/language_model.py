"""
Sparse Mixture of Experts Language Model implementation in JAX/Flax.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Optional

from .transformer_block import Block


class SparseMoELanguageModel(nn.Module):
    """
    Sparse Mixture of Experts Language Model using JAX/Flax.
    """
    vocab_size: int
    n_embd: int
    n_head: int
    num_experts: int
    top_k: int
    n_layer: int
    block_size: int
    dropout_rate: float = 0.1
    
    def setup(self):
        """Initialize the model components."""
        # Token embeddings
        self.token_embedding_table = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.n_embd
        )
        
        # Position embeddings
        self.position_embedding_table = nn.Embed(
            num_embeddings=self.block_size,
            features=self.n_embd
        )
        
        # Transformer blocks
        self.blocks = [
            Block(
                n_embd=self.n_embd,
                n_head=self.n_head,
                num_experts=self.num_experts,
                block_size=self.block_size,
                top_k=self.top_k,
                dropout_rate=self.dropout_rate
            ) for _ in range(self.n_layer)
        ]
        
        # Final layer norm
        self.ln_f = nn.LayerNorm()
        
        # Language model head
        self.lm_head = nn.Dense(self.vocab_size)

    def __call__(self, idx, targets=None, training=False, rng_key=None):
        """
        Forward pass through the model.
        
        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            targets: Target token indices of shape (batch_size, seq_len) for loss calculation
            training: Whether in training mode
            rng_key: Random key for MoE routing
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.shape
        
        # Get token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos = jnp.arange(T)
        pos_emb = self.position_embedding_table(pos)  # (T, C)
        
        # Combine token and position embeddings
        x = tok_emb + pos_emb  # (B, T, C)
        
        # Process through transformer blocks
        if rng_key is not None:
            rng_keys = jax.random.split(rng_key, self.n_layer)
        else:
            rng_keys = [None] * self.n_layer
        
        for i, block in enumerate(self.blocks):
            x = block(x, rng_key=rng_keys[i], training=training)  # (B, T, C)
        
        # Apply final layer norm
        x = self.ln_f(x)  # (B, T, C)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets are provided
        if targets is not None:
            # Reshape logits and targets for loss calculation
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            
            # Calculate cross-entropy loss
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits_flat, targets_flat
            ).mean()
        else:
            loss = None
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, rng_key=None):
        """
        Generate tokens autoregressively.
        
        Args:
            idx: Starting token indices of shape (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            rng_key: Random key for sampling
            
        Returns:
            Generated token sequence of shape (batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            
            # Get the predictions
            logits, _ = self(idx_cond, rng_key=rng_key)
            
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Sample from the distribution
            if rng_key is not None:
                rng_key, sample_key = jax.random.split(rng_key)
                next_idx = jax.random.categorical(sample_key, logits, axis=-1)  # (B,)
            else:
                # Fallback to greedy sampling if no RNG key
                next_idx = jnp.argmax(logits, axis=-1)  # (B,)
            
            # Add batch dimension if needed
            next_idx = next_idx.reshape(-1, 1)
            
            # Append sampled index to the running sequence
            idx = jnp.concatenate([idx, next_idx], axis=1)  # (B, T+1)
        
        return idx 