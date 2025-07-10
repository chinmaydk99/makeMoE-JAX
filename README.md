# Sparse Mixture of Experts Language Model - JAX Implementation

A from-scratch implementation of a Sparse Mixture of Experts (MoE) language model in JAX/Flax, ported from the original PyTorch implementation inspired by Andrej Karpathy's makemore project.

## Features

- **Sparse MoE Architecture**: Implements sparse mixture of experts with top-k routing
- **JIT Compilation**: Optimized for performance with JAX's JIT compilation
- **Modular Design**: Clean, reusable components for easy experimentation
- **Comprehensive Testing**: Full test suite ensuring correctness
- **Detailed Documentation**: Complete porting guide and API documentation

## Architecture Components

### Core Components
- **Expert Module**: Simple MLP experts with 4x expansion
- **Router**: Top-k and noisy top-k routing for load balancing
- **Sparse MoE Layer**: JIT-compatible sparse expert routing
- **Multi-Head Attention**: Causal self-attention mechanism
- **Transformer Block**: Combines attention and MoE layers
- **Language Model**: Complete autoregressive language model

### Key Improvements Over PyTorch Version
- **JIT Compatibility**: Avoided dynamic shapes for better performance
- **Functional Programming**: Leverages JAX's functional paradigm
- **Better Modularity**: Separated concerns into focused modules
- **Type Safety**: Comprehensive type annotations
- **Memory Efficiency**: Optimized memory usage patterns

## Quick Start

### Installation
```bash
pip install jax flax optax
```

### Basic Usage
```python
import jax
import jax.random as random
from src import SparseMoELanguageModel, create_train_state, train_model

# Model configuration
model_config = {
    'vocab_size': 10000,
    'n_embd': 256,
    'n_head': 8,
    'num_experts': 8,
    'top_k': 2,
    'n_layer': 4,
    'block_size': 128,
    'dropout_rate': 0.1
}

# Create training state
key = random.PRNGKey(42)
state = create_train_state(key, model_config, learning_rate=1e-3)

# Train the model (with your data)
final_state = train_model(state, train_data, num_epochs=10, batch_size=32, rng_key=key)
```

### Text Generation
```python
# Generate text
model = SparseMoELanguageModel(**model_config)
start_tokens = jnp.array([[1, 2, 3, 4]])  # Your starting tokens

generated = model.apply(
    final_state.params,
    start_tokens,
    max_new_tokens=50,
    temperature=0.8,
    rng_key=key,
    method=model.generate
)
```

## Testing

Run the comprehensive test suite:
```bash
cd makeMoE-JAX
python -m src.test_model
```

Run the Shakespeare example:
```bash
python -m src.example
```

## Project Structure

```
makeMoE-JAX/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── expert.py             # Expert MLP module
│   ├── router.py             # Top-k and noisy routing
│   ├── sparse_moe.py         # Sparse MoE layer
│   ├── attention.py          # Multi-head attention
│   ├── transformer_block.py  # Transformer block
│   ├── language_model.py     # Main language model
│   ├── training.py           # Training utilities
│   ├── test_model.py         # Test suite
│   └── example.py            # Shakespeare example
├── PORTING_GUIDE.md          # Detailed porting documentation
└── README.md                 # This file
```

## Key Differences from PyTorch

### 1. JIT Compilation Compatibility
The biggest challenge was making the sparse routing JIT-compatible. The original PyTorch version used dynamic shapes:

```python
# PyTorch (dynamic shapes)
expert_mask = (indices == i).any(dim=-1)
if expert_mask.any():
    expert_input = x[expert_mask]
```
