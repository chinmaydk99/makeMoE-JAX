"""
Example script demonstrating the Sparse MoE Language Model with Shakespeare data.
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


def create_shakespeare_data():
    """
    Create a simple character-level tokenizer and dataset.
    This is a simplified version of the original Shakespeare dataset processing.
    """
    text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream—ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause—there's the respect
    That makes calamity of so long life.
    """
    
    # Create character-level vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode the text
    data = jnp.array([stoi[c] for c in text])
    
    return data, vocab_size, stoi, itos


def train_shakespeare_model():
    """Train a model on Shakespeare-like text."""
    print("Creating Shakespeare dataset...")
    
    # Create dataset
    data, vocab_size, stoi, itos = create_shakespeare_data()
    
    # Split into train/val
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Model configuration
    model_config = {
        'vocab_size': vocab_size,
        'n_embd': 128,
        'n_head': 8,
        'num_experts': 8,
        'top_k': 2,
        'n_layer': 4,
        'block_size': 64,
        'dropout_rate': 0.1
    }
    
    # Training hyperparameters
    learning_rate = 1e-3
    num_epochs = 20
    batch_size = 16
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Model parameters: {model_config}")
    
    # Create batched training data
    def create_batched_data(data, block_size, num_samples):
        """Create batched data from the text."""
        # Randomly sample starting positions
        key = random.PRNGKey(42)
        max_start = len(data) - block_size
        starts = random.randint(key, (num_samples,), 0, max_start)
        
        # Create sequences
        sequences = []
        for start in starts:
            seq = data[start:start + block_size]
            sequences.append(seq)
        
        return jnp.stack(sequences)
    
    # Create training dataset
    train_ds = {
        'tokens': create_batched_data(train_data, model_config['block_size'], 1000)
    }
    
    val_ds = {
        'tokens': create_batched_data(val_data, model_config['block_size'], 200)
    }
    
    # Initialize model and training state
    key = random.PRNGKey(42)
    state = create_train_state(key, model_config, learning_rate)
    
    print("\nStarting training...")
    print("=" * 60)
    
    # Train the model
    final_state = train_model(state, train_ds, num_epochs, batch_size, key)
    
    print("=" * 60)
    print("Training completed!")
    
    # Evaluate on validation set
    eval_key = random.PRNGKey(123)
    val_metrics = evaluate_model(final_state, val_ds, batch_size, eval_key)
    print(f"Validation: loss = {val_metrics['loss']:.4f}, perplexity = {val_metrics['perplexity']:.4f}")
    
    return final_state, model_config, stoi, itos


def generate_text(state, model_config, stoi, itos, prompt="To be", max_tokens=100):
    """Generate text using the trained model."""
    print(f"\nGenerating text with prompt: '{prompt}'")
    
    # Encode the prompt
    prompt_tokens = jnp.array([[stoi.get(c, 0) for c in prompt]])
    
    # Generate
    key = random.PRNGKey(456)
    model = SparseMoELanguageModel(**model_config)
    
    generated_tokens = model.apply(
        state.params,
        prompt_tokens,
        max_new_tokens=max_tokens,
        temperature=0.8,
        rng_key=key,
        method=model.generate
    )
    
    # Decode the generated text
    generated_text = ''.join([itos.get(int(token), '?') for token in generated_tokens[0]])
    
    print(f"Generated text:\n{generated_text}")
    return generated_text


def compare_with_dummy_data():
    """Compare performance with dummy data vs structured text."""
    print("\n" + "=" * 60)
    print("Comparing with dummy data...")
    
    # Create dummy data
    key = random.PRNGKey(789)
    dummy_ds = create_dummy_data(key, 1000, 64, 50)
    
    # Model configuration for dummy data
    dummy_config = {
        'vocab_size': 50,
        'n_embd': 64,
        'n_head': 4,
        'num_experts': 4,
        'top_k': 2,
        'n_layer': 2,
        'block_size': 64,
        'dropout_rate': 0.1
    }
    
    # Train on dummy data
    dummy_state = create_train_state(key, dummy_config, 1e-3)
    dummy_state = train_model(dummy_state, dummy_ds, 5, 16, key)
    
    # Evaluate
    dummy_metrics = evaluate_model(dummy_state, dummy_ds, 16, key)
    print(f"Dummy data final: loss = {dummy_metrics['loss']:.4f}, perplexity = {dummy_metrics['perplexity']:.4f}")


def main():
    """Main function to run the example."""
    print("=" * 60)
    print("Sparse MoE Language Model - Shakespeare Example")
    print("=" * 60)
    
    try:
        # Train the model
        final_state, model_config, stoi, itos = train_shakespeare_model()
        
        # Generate some text
        generate_text(final_state, model_config, stoi, itos, "To be", 50)
        generate_text(final_state, model_config, stoi, itos, "The ", 50)
        
        # Compare with dummy data
        compare_with_dummy_data()
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during example: {e}")
        raise


if __name__ == "__main__":
    main() 