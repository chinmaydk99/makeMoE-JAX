"""
Sparse Mixture of Experts Language Model implementation in JAX/Flax.
"""

from .expert import Expert
from .router import TopKRouter, NoisyTopKRouter
from .sparse_moe import SparseMoE, SparseMoEOptimized
from .attention import Head, MultiHeadAttention, MultiHeadAttentionOptimized
from .transformer_block import Block
from .language_model import SparseMoELanguageModel
from .training import (
    create_train_state,
    train_step,
    train_epoch,
    train_model,
    create_dummy_data,
    evaluate_model,
    loss_fn
)

__all__ = [
    'Expert',
    'TopKRouter',
    'NoisyTopKRouter',
    'SparseMoE',
    'SparseMoEOptimized',
    'Head',
    'MultiHeadAttention',
    'MultiHeadAttentionOptimized',
    'Block',
    'SparseMoELanguageModel',
    'create_train_state',
    'train_step',
    'train_epoch',
    'train_model',
    'create_dummy_data',
    'evaluate_model',
    'loss_fn'
] 