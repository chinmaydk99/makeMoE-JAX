import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jax.random as random

class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """
    n_embd : int
    dropout_rate : float = 0.1

    @nn.compact
    def __call__(self, x, training : bool = True):
        x = nn.Dense(4 * self.n_embd, name = "w1")(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_embd, name = "w2")(x)
        x = nn.Dropout(
            rate = self.dropout_rate,
            deterministic = not training
        )(x)
        return x
    
class TopKRouter(nn.Module):
    """ Router that selects top-k experts for each token """
    n_embd : int
    num_experts : int
    top_k : int

    @nn.compact
    def __call__(self, x):
        logits = nn.Dense(self.num_experts)(x) # (batch, seq_len, num_experts)
        top_k_logits, top_k_indices = jax.lax.top_k(logits, self.top_k)
        zeros = jnp.full_like(logits, -jnp.inf)
        sparse_logits = jnp.put_along_axis(zeros, top_k_indices, top_k_logits, axis = -1, inplace = False)
        router_output = jax.nn.softmax(sparse_logits, axis = -1)
        return router_output, top_k_indices

class NoisyTopKRouter(nn.Module):
    """ Router that selects top-k experts for each token with noise """
    n_embd : int
    num_experts : int
    top_k : int

    @nn.compact
    def __call__(self, x, rng_key):
        logits = nn.Dense(self.num_experts)(x) # (batch, seq_len, num_experts)
        noise_logits = nn.Dense(self.num_experts)(x) # (batch, seq_len, num_experts)

        noise = jax.random.normal(rng_key, logits.shape)
        noise = noise * jax.nn.softplus(noise_logits)
        logits = logits + noise

        top_k_logits, top_k_indices = jax.lax.top_k(logits, self.top_k)
        zeros = jnp.full_like(logits, -jnp.inf)
        sparse_logits = jnp.put_along_axis(zeros, top_k_indices, top_k_logits, axis = -1, inplace = False)
        router_output = jax.nn.softmax(sparse_logits, axis = -1)
        return router_output, top_k_indices

class SparseMoE(nn.Module):
    """Sparse Mixture of Experts using JAX's batching capabilities"""
    n_embd: int
    num_experts: int
    top_k: int
    
    def setup(self):
        self.router = NoisyTopKRouter(
            n_embd=self.n_embd,
            num_experts=self.num_experts,
            top_k=self.top_k
        )
        self.experts = [Expert(n_embd=self.n_embd) for _ in range(self.num_experts)]
    
    def __call__(self, x, training=False, rng_key=None):
        # Get shape of input
        B, T, C = x.shape

        # Get routing probabilities and expert indices
        router_weights, router_indices = self.router(x, rng_key=rng_key) # ( batch, seq_len, num_experts)
        
        # Initialize the output tensor upfront - don't check if it exists later
        final_output = jnp.zeros_like(x) # (batch, seq_len, n_embd)
        
        # Flattening the input and router weights and indices
        x_reshaped = x.reshape(-1, C) # (batch * seq_len, n_embd)
        router_weights_flat = router_weights.reshape(-1, self.num_experts) # (batch * seq_len, num_experts)
        router_indices_flat = router_indices.reshape(-1, self.top_k) # (batch * seq_len, top_k)
        
        # Process each expert
        for e in range(self.num_experts):
            # Create mask for tokens going to this expert
            expert_mask = (router_indices == e).any(axis=-1) # (batch, seq_len)
            
            # Skip if no tokens assigned to this expert
            if not jnp.any(expert_mask):
                continue
                
            # Get positions where this expert is used
            token_indices = jnp.where(expert_mask) # (num_tokens, 2)
            batch_idx, seq_idx = token_indices
            
            # Flatten indices for easier extraction
            # Converts (batch_idx, seq_idx) to a 1D index
            flat_indices = batch_idx * T + seq_idx # (num_tokens)
            
            # Extract inputs for this expert
             # Dispatch step : Only gathering tokens that need to be processed by this expert
            expert_inputs = x_reshaped[flat_indices] # (num_tokens, n_embd)
            
            # Process through the expert
            expert_outputs = self.experts[e](expert_inputs, training=training) # (num_tokens, n_embd)
            
            # Get corresponding weights
            expert_weights = router_weights[batch_idx, seq_idx, e]
            
            # Weight the outputs
            weighted_outputs = expert_outputs * expert_weights[:, None]
            
            # Use functional update to add to final output
            for i, (b_idx, s_idx) in enumerate(zip(batch_idx, seq_idx)):
                final_output = final_output.at[b_idx, s_idx].add(weighted_outputs[i])
        
        return final_output

class Head(nn.Module):
    """ one head of self-attention """
    head_size: int
    n_embd: int
    block_size: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training : bool = False):
        B, T, C = x.shape

        # Create key, query, and value projections
        key = nn.Dense(self.head_size, use_bias=False, name ="key")(x)
        query = nn.Dense(self.head_size, use_bias=False, name ="query")(x)
        value = nn.Dense(self.head_size, use_bias=False, name ="value")(x)

        # Compute attention scores
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = query @ jnp.swapaxes(key, -2, -1) * (self.head_size**-0.5)
        
        # Create causal mask for the current block
        T_use = min(T, self.block_size)
        mask = jnp.tril(jnp.ones((T_use, T_use)))
        mask = jnp.broadcast_to(mask, (x.shape[0], T_use, T_use)) # (T_use, T_use) -> (B, T_use, T_use)
        
        # Apply causal mask
        wei = jnp.where(mask, wei, -jnp.inf)

        wei = jax.nn.softmax(wei, axis = -1)
        wei = nn.Dropout(self.dropout_rate, deterministic = not training)(wei)

        out = wei @ value
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    n_embd: int
    n_head: int
    head_size: int
    block_size: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training=False):
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
    
class Block(nn.Module):
    n_embd : int
    n_head : int
    num_experts : int
    block_size : int
    top_k : int

    def setup(self):
        self.sa = MultiHeadAttention(
            n_embd = self.n_embd,
            n_head = self.n_head,
            head_size = self.n_embd // self.n_head,
            block_size = self.block_size)
        
        self.smoe = SparseMoE(
            n_embd = self.n_embd,
            num_experts = self.num_experts,
            top_k = self.top_k
        )

        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
    
    def __call__(self, x, rng_key, training : bool = False):
        sa_key, moe_key = jax.random.split(rng_key)

        # Self attention with residual connection
        x = x + self.sa(self.ln1(x), training = training)

        # Sparse MoE with residual connection
        x = x + self.smoe(self.ln2(x), training = training, rng_key = moe_key)

        return x