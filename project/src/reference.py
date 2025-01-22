# torch equivalent:
from jax import numpy as jnp
import jax
import torch
import math


@jax.jit
def scaled_dot_product_solution(q, k, v, mask=None):
    """
    PLEASE DON'T LOOK AT THIS THIS IS JUST USED FOR TESTING 
    """
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = jax.nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values

@jax.jit
def sdpa_with_mha_and_mask_solution(q, k, v, mask=None, num_heads=8):
    B, N, D = q.shape
    head_dim = D // num_heads
    # split into heads
    q = q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    # apply scaled dot product attention
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = jax.nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    # reshape back to original shape
    return values.reshape(B, N, D)

@jax.jit
def triangle_attn_solution(q, k, v):
    B, N, N, D = q.shape
    # O(n**3) triangle attention on starting node
    q = q.reshape(B, N, N, D)
    k = k.transpose(0, 1, 3, 2)
    v = v.transpose(0, 1, 3, 2)
    attn_out = scaled_dot_product_solution(q, k, v)
    return attn_out.reshape(B, N, D)

def triangle_attn(q, k, v):
    # Q3
    # Implement the triangle attention forward pass: 
    ...
    return ...
