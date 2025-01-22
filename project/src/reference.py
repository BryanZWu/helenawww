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
        # Reshape mask to [B, 1, N, N] to broadcast across heads
        mask = mask[:, None, :, :] if mask.ndim == 3 else mask
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = jax.nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    # reshape back to original shape
    return values.reshape(B, N, D)

@jax.jit
def triangle_attn_solution(q, k, v):
    B, N, N, D = q.shape
    # O(n**3) triangle attention on starting node
    q = q.reshape(B, N, N, D) # B, N_f, N_t, D
    k = k.transpose(0, 1, 3, 2) # B, N_f, D, N_t
    v = v.transpose(0, 1, 2, 3) # B, N_f, N_t, D

    # B, N_f, N_t, N_t: each N_f, N_t dim attends
    # to each of N_t nodes that also start at N_f
    attn_logits = jnp.matmul(q, k)
    attn_score = jax.nn.softmax(attn_logits, axis=-1)
    attn_out = jnp.matmul(attn_score, v) # B, N_f, N_t, D
    return attn_out
