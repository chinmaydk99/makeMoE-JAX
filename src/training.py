"""
Training utilities for the Sparse MoE Language Model.
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Dict, Any
import time

from .language_model import SparseMoELanguageModel


def create_train_state(key, model_config: Dict[str, Any], learning_rate: float):
    """
    Create initial training state for the SparseMoELanguageModel.
    
    Args:
        key: Random key for initialization
        model_config: Model configuration dictionary
        learning_rate: Learning rate for optimizer
        
    Returns:
        Training state
    """
    model = SparseMoELanguageModel(**model_config)
    
    # Create dummy input for initialization
    dummy_tokens = jnp.ones((1, model_config['block_size']), dtype=jnp.int32)
    
    # Initialize model parameters
    params = model.init(
        {'params': key, 'dropout': jax.random.PRNGKey(0)},
        dummy_tokens,
        training=False,
        rng_key=key
    )
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )


def loss_fn(logits, targets):
    """
    Calculate cross-entropy loss for the language model.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        targets: Target token indices of shape (batch_size, seq_len)
        
    Returns:
        Cross-entropy loss
    """
    # Shifting targets and logits for next-token prediction
    shifted_targets = targets[:, 1:]  # Remove first token
    shifted_logits = logits[:, :-1]   # Remove last prediction
    
    # Flattening for cross entropy
    shifted_targets = shifted_targets.reshape(-1)
    shifted_logits = shifted_logits.reshape(-1, logits.shape[-1])
    
    # Cross-entropy loss
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
        shifted_logits, shifted_targets
    ))


@jax.jit
def train_step(state, batch, dropout_rng):
    """
    Single training step for the MoE model.
    
    Args:
        state: Training state
        batch: Batch of data containing 'tokens'
        dropout_rng: Random key for dropout
        
    Returns:
        Updated state and metrics
    """
    
    def loss_fn_internal(params):
        logits, loss = state.apply_fn(
            params,
            batch['tokens'],
            targets=batch['tokens'],  # Use same tokens as targets for next-token prediction
            training=True,
            rng_key=dropout_rng,
            rngs={'dropout': dropout_rng}
        )
        return loss, logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn_internal, has_aux=True)(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    # Calculate perplexity (standard language model metric)
    perplexity = jnp.exp(loss)
    
    return state, {'loss': loss, 'perplexity': perplexity}


def train_epoch(state, train_ds, batch_size, epoch_rng):
    """
    Train for one epoch.
    
    Args:
        state: Training state
        train_ds: Training dataset
        batch_size: Batch size
        epoch_rng: Random key for epoch
        
    Returns:
        Updated state and epoch metrics
    """
    train_ds_size = len(train_ds['tokens'])
    steps_per_epoch = train_ds_size // batch_size
    
    # Create permutation for shuffling
    perms = jax.random.permutation(epoch_rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    
    epoch_metrics = []
    
    for perm in perms:
        batch = {
            'tokens': train_ds['tokens'][perm]
        }
        dropout_rng = jax.random.fold_in(epoch_rng, state.step)
        state, metrics = train_step(state, batch, dropout_rng)
        epoch_metrics.append(metrics)
    
    # Compute average metrics
    epoch_metrics = {
        'loss': jnp.mean(jnp.stack([m['loss'] for m in epoch_metrics])),
        'perplexity': jnp.mean(jnp.stack([m['perplexity'] for m in epoch_metrics]))
    }
    
    return state, epoch_metrics


def train_model(state, train_ds, num_epochs, batch_size, rng_key):
    """
    Complete training loop for the MoE model.
    
    Args:
        state: Initial training state
        train_ds: Training dataset
        num_epochs: Number of epochs to train
        batch_size: Batch size
        rng_key: Random key
        
    Returns:
        Final training state
    """
    
    for epoch in range(num_epochs):
        # Get a new RNG key for this epoch
        rng_key, epoch_key = jax.random.split(rng_key)
        
        # Train for one epoch
        start_time = time.time()
        state, train_metrics = train_epoch(state, train_ds, batch_size, epoch_key)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:2d}: "
              f"loss = {train_metrics['loss']:.4f}, "
              f"perplexity = {train_metrics['perplexity']:.4f}, "
              f"time = {epoch_time:.2f}s")
    
    return state


def create_dummy_data(key, num_samples, seq_len, vocab_size):
    """
    Create dummy token data for testing.
    
    Args:
        key: Random key
        num_samples: Number of samples
        seq_len: Sequence length
        vocab_size: Vocabulary size
        
    Returns:
        Dictionary with 'tokens' key
    """
    tokens = jax.random.randint(key, (num_samples, seq_len), 0, vocab_size)
    return {'tokens': tokens}


def evaluate_model(state, eval_ds, batch_size, rng_key):
    """
    Evaluate the model on a dataset.
    
    Args:
        state: Training state
        eval_ds: Evaluation dataset
        batch_size: Batch size
        rng_key: Random key
        
    Returns:
        Evaluation metrics
    """
    eval_ds_size = len(eval_ds['tokens'])
    steps = eval_ds_size // batch_size
    
    # Create batches
    perms = jnp.arange(steps * batch_size).reshape((steps, batch_size))
    
    losses = []
    
    for perm in perms:
        batch = {
            'tokens': eval_ds['tokens'][perm]
        }
        
        # Forward pass without training
        logits, loss = state.apply_fn(
            state.params,
            batch['tokens'],
            targets=batch['tokens'],
            training=False,
            rng_key=rng_key
        )
        
        losses.append(loss)
    
    avg_loss = jnp.mean(jnp.stack(losses))
    perplexity = jnp.exp(avg_loss)
    
    return {'loss': avg_loss, 'perplexity': perplexity} 