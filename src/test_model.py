"""
Test script for the Sparse MoE Language Model implementation.
"""

import jax
import jax.numpy as jnp
import jax.random as random
from . import (
    SparseMoELanguageModel,
    create_train_state,
    train_model,
    create_dummy_data,
    evaluate_model
)


def test_model_components():
    """Test individual components of the model."""
    print("Testing model components...")
    
    # Set up a consistent RNG key
    main_key = random.PRNGKey(42)
    
    # Test parameters - small values for quick testing
    batch_size = 4
    seq_len = 16
    vocab_size = 100
    embed_dim = 64
    n_head = 4
    num_experts = 4
    top_k = 2
    n_layer = 2
    block_size = 32
    
    # Model configuration
    model_config = {
        'vocab_size': vocab_size,
        'n_embd': embed_dim,
        'n_head': n_head,
        'num_experts': num_experts,
        'top_k': top_k,
        'n_layer': n_layer,
        'block_size': block_size,
        'dropout_rate': 0.1
    }
    
    # Create dummy input data
    main_key, input_key = random.split(main_key)
    dummy_tokens = random.randint(input_key, (batch_size, seq_len), 0, vocab_size)
    
    # Initialize the model
    model = SparseMoELanguageModel(**model_config)
    
    # Initialize parameters
    main_key, init_key, dropout_key = random.split(main_key, 3)
    model_params = model.init(
        {'params': init_key, 'dropout': dropout_key},
        dummy_tokens,
        training=True,
        rng_key=init_key
    )
    
    # Forward pass with loss
    main_key, forward_key = random.split(main_key)
    logits, loss = model.apply(
        model_params,
        dummy_tokens,
        targets=dummy_tokens,
        training=True,
        rng_key=forward_key,
        rngs={'dropout': dropout_key}
    )
    
    print(f"  Model logits shape: {logits.shape} (expected: {(batch_size, seq_len, vocab_size)})")
    print(f"  Loss value: {loss}")
    
    # Test generation
    main_key, gen_key = random.split(main_key)
    start_tokens = random.randint(gen_key, (1, 4), 0, vocab_size)
    
    generated = model.apply(
        model_params,
        start_tokens,
        max_new_tokens=5,
        temperature=0.8,
        rng_key=gen_key,
        method=model.generate
    )
    
    print(f"  Generated sequence shape: {generated.shape} (expected: {(1, 4+5)})")
    print(f"  Generated tokens: {generated[0]}")
    
    return True


def test_training_loop():
    """Test the training loop."""
    print("\nTesting training loop...")
    
    # Set up a consistent RNG key
    main_key = random.PRNGKey(42)
    
    # Model configuration - smaller for faster testing
    model_config = {
        'vocab_size': 100,
        'n_embd': 64,
        'n_head': 4,
        'num_experts': 4,
        'top_k': 2,
        'n_layer': 2,
        'block_size': 32,
        'dropout_rate': 0.1
    }
    
    # Training hyperparameters
    learning_rate = 1e-3
    num_epochs = 3
    batch_size = 16
    
    # Create dummy data
    main_key, data_key = random.split(main_key)
    train_ds = create_dummy_data(
        data_key, 
        num_samples=200, 
        seq_len=model_config['block_size'], 
        vocab_size=model_config['vocab_size']
    )
    
    # Initialize model and training state
    main_key, init_key = random.split(main_key)
    state = create_train_state(init_key, model_config, learning_rate)
    
    print("Starting training...")
    print("-" * 40)
    
    # Train the model
    final_state = train_model(state, train_ds, num_epochs, batch_size, main_key)
    
    print("-" * 40)
    print("Training completed!")
    
    # Test evaluation
    main_key, eval_key = random.split(main_key)
    eval_metrics = evaluate_model(final_state, train_ds, batch_size, eval_key)
    print(f"Final evaluation: loss = {eval_metrics['loss']:.4f}, perplexity = {eval_metrics['perplexity']:.4f}")
    
    return final_state


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Sparse MoE Language Model (JAX Implementation)")
    print("=" * 60)
    
    try:
        # Test model components
        test_model_components()
        
        # Test training loop
        test_training_loop()
        
        print("\n" + "=" * 60)
        print("All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        raise


if __name__ == "__main__":
    main() 