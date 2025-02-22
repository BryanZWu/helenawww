# torch equivalent:
from jax import numpy as jnp
import jax
import math
from reference import scaled_dot_product_solution, sdpa_with_mha_and_mask_solution, triangle_attn_solution, triangle_attention_with_mha_solution

B, N, D = 2, 256, 128
print("JAX devices:", jax.devices())

@jax.jit
def vanilla_attention_forward(q, k, v):
    # Implement the vanilla attention forward pass: 
    # attn_out = softmax(Q @ K^T) / sqrt(D)
    ...
    D = q.shape[-1] 
    # k dimension: (2, 256, 128), we want 2, 128, 256
    transposed_k = jnp.transpose(k, (0, 2, 1)) 
    softmax = jax.nn.softmax(q @ transposed_k  / math.sqrt(D), axis=2) # for each query, the similar scores for all keys should sum to 1
    out = jnp.matmul(softmax, v)# apply to v
    return out

@jax.jit
def sdpa_with_mha_and_mask(q, k, v, mask=None):
    """
    Q2: two really important items in attention is the mask and 
    multi-head attention. 
    The mask is a (B, N, N) tensor where each element is either 0 or 1,
    for whether the node is allowed to "look up" another node. THis is 
    useful in autoregresive modeling (where a word can only see words before 
    it in the sequence).
    Multi-head attention is simply having different sets of QKV that can 
    each compute its own attention matrix and handle a different concept.
    """

    # For this problem, assume each head has 64 dims
    B, N, D = q.shape
    # Assume D is divisible by num_heads, common in transformer architectures
    # This makes num_heads implicit in the input shape
    num_heads = D // 64  # standard head_dim is usually 64
    head_dim = D // num_heads
    assert head_dim * num_heads == D

    D = q.shape[-1] # split d by num of heads
    # k dimension: (2, 256, 128), we want 2, 128, 256
    # convert BxNxD to Bxnum_headsxNxquery_dim, where query_dim = D/num_heads
    query_dim = D//num_heads
    
    q = q.reshape(B, N, num_heads, query_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, N, num_heads, query_dim).transpose(0, 2, 1, 3) #  Bxnum_headsxNxquery_dim
    attn = q @ k.transpose(0, 1, 3, 2)  #  K.t should be Bxnum_headsxquery_dimxN
    # attn shape hiiii (2, 2, 256, 256)
    mask_expanded = jnp.expand_dims(mask, axis=1)
    masked = jnp.where(mask_expanded == 0, -9e15, attn) # apply mask, cant be multipication since 0 matters in softmax
    softmax = jax.nn.softmax(masked / math.sqrt(query_dim), axis=-1) # for each query, the similar scores for all keys should sum to 1
    v_reshaped = v.reshape(B, N, num_heads, query_dim) 
    v_transpose = v_reshaped.transpose(0, 2, 1, 3)
    out = jnp.matmul(softmax, v_transpose)# apply to v

    return out.reshape(B, N, D)

def triangle_attn(q, k, v):
    # Q3
    # Implement the triangle attention forward pass: 
    # Q K V are each of shape (B, N, N, D)
    # For this first part, let's only do triangle attention on
    # the starting node

    # Compute attention scores. The attention score should be 
    # of shape (B, N, N, N) = (B, N_f, N_t, N_t) where N_f is the 
    # "from" node and N_t is the "to" node.
    # It logically translates to "For each item B in the batch, for each
    # from(N_f)-to(N_t) pair, here is a N_t attention score corresponding "
    # to each of the N_t nodes that also start at N_f.
    D = q.shape[-1] 
    # expanded_q = jnp.expand_dims(q, axis=2)  # Shape becomes (b, n, 1, d)
    # expanded_q = jnp.broadcast_to(expanded_q, (B, N, N, D))  # Shape becomes (b, n, n, d)
    # expanded_k = jnp.expand_dims(k, axis=2)  
    # expanded_k = jnp.broadcast_to(expanded_k, (B, N, N, D)) 
    transposed_k = jnp.transpose(k, (0, 1, 3, 2)) 
    # Apply softmax to the attention scores
    print("hhhhhhhh", (q @ transposed_k).shape)
    softmax = jax.nn.softmax(q @ transposed_k  / math.sqrt(D), axis=-1) # for each query, the similar scores for all keys should sum to 1
    # Apply the attention scores to the values
    # This logically translates to "For each item B in the batch, for each
    # from(N_f)-to(N_t) pair, apply the N_t attention scores to the N_t values
    # that also start at N_f.
    # softmax = jnp.transpose(softmax, (0, 3, 1, 2)) 
    out = jnp.matmul(softmax, v)# apply to v
    # Return the (B, N, N, D) output
    return out

@jax.jit
def triangle_attn_with_mha(q, k, v, from_starting_node=True):
    """Q4: Triangle Attention with Multi-Head Attention
    
    This is an extension of Q3 where we add multi-head attention to triangle attention.
    """
    B, N, N, D = q.shape
    head_dim = 64
    num_heads = D // head_dim
    assert head_dim * num_heads == D

    # Your previous work saved here
    # query_dim = D//num_heads
    # q = q.reshape(B, N, N, num_heads, query_dim).transpose(0, 3, 1, 2, 4)
    # k = k.reshape(B, N, num_heads, query_dim).transpose(0, 2, 1, 3) #  B x num_heads x N x query_dim
    # attn = q @ k.transpose(0, 1, 3, 2)  #  K.t should be Bxnum_headsxquery_dimxN
    # mask_expanded = jnp.expand_dims(mask, axis=1)
    # masked = attn + mask_expanded # apply mask, cant be multipication since 0 matters in softmax
    # softmax = jax.nn.softmax(masked / math.sqrt(D), axis=-1) # for each query, the similar scores for all keys should sum to 1
    # v_reshaped = v.reshape(B, N, num_heads, query_dim) 
    # v_transpose = v_reshaped.transpose(0, 2, 1, 3)
    # out = jnp.matmul(softmax, v_transpose)# apply to v
    # return out
    # End prev work

    # Pull out the head dimension from D, and combine it with the batch dimension
    ...

    # Triangle attention matmul is slightly different between from_starting_node and not
    def qkv_shapes_from_starting_node(args):
        q, k, v = args
        ...
        return q, k, v
    def qkv_shapes_from_ending_node(args):
        q, k, v = args
        ...
        return q, k, v
    q, k, v = jax.lax.cond(
        from_starting_node,
        qkv_shapes_from_starting_node,
        qkv_shapes_from_ending_node,
        (q, k, v)
    )
    
    # Create attention logits
    attn_logits = ...
    # apply softmax
    attn_score = ...

    # apply to values
    attn_out = ...
    
    # Now that matmul is done, restore the (B * n_heads, n_from, n_to, head_dim) ordering
    attn_out = jax.lax.cond(
        from_starting_node,
        lambda x: ...,
        lambda x: ...,
        attn_out,
    )

    # Pull out the n heads and put it back into D
    attn_out = ...

    
    return attn_out

def main(): 
    # First, make sure that the vanilla attention forward pass works
    # is it this command pip install -U "jax[cuda12]" torch? ohkay

    B, N, N, D = 2, 256, 256, 128

    key = jax.random.PRNGKey(0)
    q = jax.random.uniform(key, (B, N, D))
    k = jax.random.uniform(key, (B, N, D))
    v = jax.random.uniform(key, (B, N, D))
    # print(q, k, v)
    expected1 = scaled_dot_product_solution(q, k, v)
    out1 = vanilla_attention_forward(q, k, v)
    assert jnp.allclose(out1, expected1)
    print("Q1 is good!")

    # Q2
    mask = jax.random.uniform(key, (B, N, N)) > 0.5
    out2 = sdpa_with_mha_and_mask(q, k, v, mask=mask)
    expected2 = sdpa_with_mha_and_mask_solution(q, k, v, mask=mask)
    # print(out2, expected2)
    assert jnp.allclose(out2, expected2)
    print("Q2 is good!")

    # Q3
    q_triangle = jax.random.uniform(key, (B, N, N, D))
    k_triangle = jax.random.uniform(key, (B, N, N, D))
    v_triangle = jax.random.uniform(key, (B, N, N, D))
    out3 = triangle_attn(q_triangle, k_triangle, v_triangle)
    expected3 = triangle_attn_solution(q_triangle, k_triangle, v_triangle)
    assert jnp.allclose(out3, expected3)
    print("Q3 is good!")

    expected4 = triangle_attn_with_mha_solution(q_triangle, k_triangle, v_triangle)
    out4 = triangle_attn_with_mha(q_triangle, k_triangle, v_triangle)
    assert jnp.allclose(out4, expected4)
    print("Q4 is good!")

    expected5 = triangle_attn_with_mha_solution(q_triangle, k_triangle, v_triangle, from_starting_node=False)
    out5 = triangle_attn_with_mha(q_triangle, k_triangle, v_triangle, from_starting_node=False)
    assert jnp.allclose(out5, expected5)
    print("Q5 is good!")

    
    return


    # Input: (B, N, N, D)
    in_pw_tensor = jnp.random.rand(B, N, N, D)
    # Output: (B, N, N, D)
    out = attention_forward(x)
    return out

if __name__ == "__main__":
    main()