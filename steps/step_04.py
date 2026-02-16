# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Step 04: Multi-head Attention

Implement multi-head attention that splits Q/K/V into multiple heads,
computes attention in parallel for each head, and merges the results.

Tasks:
1. Import required modules (math, F, Tensor, Linear, Module, etc.)
2. Create c_attn and c_proj linear layers
3. Implement _split_heads: reshape and transpose to add head dimension
4. Implement _merge_heads: transpose and reshape to remove head dimension
5. Implement _attn: compute attention for all heads in parallel
6. Implement forward pass: project -> split -> attend -> merge -> project

Run: pixi run s04
"""

# TODO: Import required modules
# Hint: You'll need math for scaling
# Hint: You'll need functional as F from max.nn
# Hint: You'll need Tensor, Device, DType from max.tensor and max.driver
# Hint: You'll need Dim, DimLike from max.graph
# Hint: You'll also need Linear and Module from max.nn

import math
from typing import cast
import max.functional as F

from max.tensor import Tensor
from max.nn import Linear, Module
from step_01 import GPT2Config

# TODO: Copy causal_mask function from solution_02.py
# This is the same function you implemented in Step 02
from step_03 import causal_mask

class GPT2MultiHeadAttention(Module):
    """Multi-head attention for GPT-2."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()

        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        # TODO: Create combined Q/K/V projection
        # Hint: Use Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_attn = Linear(self.embed_dim, 3 * self.embed_dim, bias=True)

        # TODO: Create output projection
        # Hint: Use Linear(self.embed_dim, self.embed_dim, bias=True)
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(
        self, tensor: Tensor, num_heads: int, attn_head_size: int
    ) -> Tensor:
        """Split the last dimension into (num_heads, head_size).

        Args:
            tensor: Input tensor, shape [batch, seq_length, n_embd]
            num_heads: Number of attention heads
            attn_head_size: Dimension of each head

        Returns:
            Tensor with shape [batch, num_heads, seq_length, head_size]
        """
        # TODO: Add head dimension
        # Hint: new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        # Hint: tensor = tensor.reshape(new_shape)
        new_shape = tensor.shape[:-1] + [num_heads, attn_head_size]
        tensor = tensor.reshape(new_shape)

        # TODO: Move heads dimension to position 1
        # Hint: return tensor.transpose(-3, -2)
        return tensor.transpose(-3, -2)

    def _merge_heads(
        self, tensor: Tensor, num_heads: int, attn_head_size: int
    ) -> Tensor:
        """Merge attention heads back to original shape.

        Args:
            tensor: Input tensor, shape [batch, num_heads, seq_length, head_size]
            num_heads: Number of attention heads
            attn_head_size: Dimension of each head

        Returns:
            Tensor with shape [batch, seq_length, n_embd]
        """
        # TODO: Move heads dimension back
        # Hint: tensor = tensor.transpose(-3, -2)
        tensor = tensor.transpose(-3, -2)

        # TODO: Flatten head dimensions
        # Hint: new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]
        # Hint: return tensor.reshape(new_shape)
        new_shape = tensor.shape[:-2] + [num_heads * attn_head_size]
        return tensor.reshape(new_shape)

    def _attn(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Compute attention for all heads in parallel.

        Args:
            query: Query tensor, shape [batch, num_heads, seq_length, head_size]
            key: Key tensor, shape [batch, num_heads, seq_length, head_size]
            value: Value tensor, shape [batch, num_heads, seq_length, head_size]

        Returns:
            Attention output, shape [batch, num_heads, seq_length, head_size]
        """
        # TODO: Implement attention computation
        # The same 5-step process: scores, scale, mask, softmax, weighted sum
        # Hint: Compute attention scores: query @ key.transpose(-1, -2)
        # Hint: Scale by sqrt(head_dim): attn_weights / math.sqrt(head_dim)
        # Hint: Apply causal mask using causal_mask function
        # Hint: Apply softmax: F.softmax(attn_weights)
        # Hint: Weighted sum: attn_weights @ value
        attn_weights = query @ key.transpose(-1, -2)
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        mask = causal_mask(query.shape[-2], 0, dtype=query.dtype, device=query.device)
        attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights)
        weighted_sum = attn_weights @ value
        return weighted_sum

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply multi-head attention.

        Args:
            hidden_states: Input tensor, shape [batch, seq_length, n_embd]

        Returns:
            Attention output, shape [batch, seq_length, n_embd]
        """
        # TODO: Project to Q, K, V
        # Hint: qkv = self.c_attn(hidden_states)
        # Hint: query, key, value = F.split(qkv, [self.split_size, self.split_size, self.split_size], axis=-1)
        qkv = self.c_attn(hidden_states)
        query, key, value = F.split(qkv, [self.split_size, self.split_size, self.split_size], axis=-1)

        # TODO: Split into multiple heads
        # Hint: query = self._split_heads(query, self.num_heads, self.head_dim)
        # Hint: key = self._split_heads(key, self.num_heads, self.head_dim)
        # Hint: value = self._split_heads(value, self.num_heads, self.head_dim)
        query = self._split_heads(cast(Tensor, query), self.num_heads, self.head_dim)
        key = self._split_heads(cast(Tensor, key), self.num_heads, self.head_dim)
        value = self._split_heads(cast(Tensor, value), self.num_heads, self.head_dim)

        # TODO: Apply attention
        # Hint: attn_output = self._attn(query, key, value)
        attn_output = self._attn(query, key, value)

        # TODO: Merge heads back
        # Hint: attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        # TODO: Output projection
        # Hint: attn_output = self.c_proj(attn_output)
        # Hint: return attn_output
        attn_output = self.c_proj(attn_output)
        return attn_output
