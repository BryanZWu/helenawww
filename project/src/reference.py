# torch equivalent:
from jax import numpy as jnp
import jax
import math
from functools import partial


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
def sdpa_with_mha_and_mask_solution(q, k, v, mask=None):
    B, N, D = q.shape
    # Assume D is divisible by num_heads, common in transformer architectures
    # This makes num_heads implicit in the input shape
    num_heads = D // 64  # standard head_dim is usually 64
    head_dim = D // num_heads
    assert head_dim * num_heads == D
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

@jax.jit
def triangle_attention_with_mha_solution(q, k, v, from_starting_node=True):
    B, N, N, D = q.shape
    head_dim = 64
    num_heads = D // head_dim
    assert head_dim * num_heads == D
    
    def qkv_shapes_from_starting_node(args):
        q, k, v = args
        q = q.reshape(B, N, N, num_heads, head_dim).transpose(0, 4, 1, 2, 3)
        k = k.reshape(B, N, N, num_heads, head_dim).transpose(0, 4, 1, 3, 2)
        v = v.reshape(B, N, N, num_heads, head_dim).transpose(0, 4, 1, 2, 3)
        return q, k, v
    
    def qkv_shapes_from_ending_node(args):
        q, k, v = args
        q = q.reshape(B, N, N, num_heads, head_dim).transpose(0, 4, 2, 1, 3)
        k = k.reshape(B, N, N, num_heads, head_dim).transpose(0, 4, 2, 3, 1)
        v = v.reshape(B, N, N, num_heads, head_dim).transpose(0, 4, 2, 1, 3)
        return q, k, v

    q, k, v = jax.lax.cond(
        from_starting_node,
        qkv_shapes_from_starting_node,
        qkv_shapes_from_ending_node,
        (q, k, v)
    )

    q = q.reshape(B * num_heads, N, N, head_dim)
    k = k.reshape(B * num_heads, N, head_dim, N)
    v = v.reshape(B * num_heads, N, N, head_dim)


    # B * H, N_f, N_t, N_t if from_starting_node else B * H, N_t, N_f, N_f
    attn_logits = jnp.matmul(q, k)
    attn_score = jax.nn.softmax(attn_logits, axis=-1)

    # B * H, N_f, N_t, head_dim if from_starting_node else B * H, N_t, N_f, head_dim
    attn_out = jnp.matmul(attn_score, v)
    attn_out = attn_out.reshape(B, num_heads, N, N, head_dim)
    
    def true_transpose(x):
        return x.transpose(0, 2, 3, 1, 4)
    
    def false_transpose(x):
        return x.transpose(0, 3, 2, 1, 4)
    
    attn_out = jax.lax.cond(
        from_starting_node,
        true_transpose,
        false_transpose,
        attn_out
    )
    
    attn_out = attn_out.reshape(B, N, N, D)
    return attn_out


if __name__ == "__main__":
    B, N, N, D = 2, 256, 256, 128

    key = jax.random.PRNGKey(0)
    q = jax.random.uniform(key, (B, N, D))
    k = jax.random.uniform(key, (B, N, D))
    v = jax.random.uniform(key, (B, N, D))
    expected1 = scaled_dot_product_solution(q, k, v)

    # Q2
    mask = jax.random.uniform(key, (B, N, N)) > 0.5
    expected2 = sdpa_with_mha_and_mask_solution(q, k, v, mask=mask)

    # Q3
    q_triangle = jax.random.uniform(key, (B, N, N, D))
    k_triangle = jax.random.uniform(key, (B, N, N, D))
    v_triangle = jax.random.uniform(key, (B, N, N, D))
    expected3 = triangle_attn_solution(q_triangle, k_triangle, v_triangle)

    # Q4
    expected4 = triangle_attention_with_mha_solution(q_triangle, k_triangle, v_triangle)
    expected5 = triangle_attention_with_mha_solution(q_triangle, k_triangle, v_triangle, from_starting_node=False)