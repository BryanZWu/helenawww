# Verifying the correctness of the custom triangle attention kernel
from jax import lax, jit, numpy as jnp
import jax
import math
@jax.jit
def triangle_attention_with_mha_solution(q, k, v, attention_mask, from_starting_node=True):
    """
    This is the reference implementation of the triangle attention with MHA,
    against which we will verify the correctness of our custom kernel.
    
    q: B, N, N, D
    k: B, N, N, D
    v: B, N, N, D
    attention_mask: (B * num_heads, N, N, N). Potentially add support for (N, N, N) later.
    """
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
    attn_logits = attn_logits / math.sqrt(head_dim)

    # apply attention mask
    attn_logits = jnp.where(attention_mask == 0, -9e15, attn_logits)
    
    # softmax
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

