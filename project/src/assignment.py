# torch equivalent:
from jax import numpy as jnp
import jax
import math
from reference import scaled_dot_product_solution, sdpa_with_mha_and_mask_solution, triangle_attn_solution

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
    softmax = jax.nn.softmax(q @ transposed_k  / math.sqrt(D), axis=-1) # for each query, the similar scores for all keys should sum to 1
    out = jnp.matmul(softmax, v)# apply to v
    return out

def sdpa_with_mha_and_mask(q, k, v, mask, num_heads):
    # Q2: two really important items in attention is the mask and 
    # multi-head attention. 
    # The mask is a (B, N, N) tensor where each element is either 0 or 1,
    # for whether the node is allowed to "look up" another node. THis is 
    # useful in autoregresive modeling (where a word can only see words before 
    # it in the sequence).
    # Multi-head attention is simply having different sets of QKV that can 
    # each compute its own attention matrix and handle a different concept.
    ...

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
    ...

    # Apply softmax to the attention scores
    ...

    # Apply the attention scores to the values
    # This logically translates to "For each item B in the batch, for each
    # from(N_f)-to(N_t) pair, apply the N_t attention scores to the N_t values
    # that also start at N_f.
    ...

    # Return the (B, N, N, D) output
    ...

    return ...
def main(): 
    # First, make sure that the vanilla attention forward pass works
    # is it this command pip install -U "jax[cuda12]" torch? ohkay

    B, N, N, D = 2, 256, 256, 128

    key = jax.random.PRNGKey(0)
    q = jax.random.uniform(key, (B, N, D))
    k = jax.random.uniform(key, (B, N, D))
    v = jax.random.uniform(key, (B, N, D))
    print(q, k, v)
    expected1 = scaled_dot_product_solution(q, k, v)
    out1 = vanilla_attention_forward(q, k, v)
    assert jnp.allclose(out1, expected1)
    print("Q1 is good!")

    # Q2
    mask = jax.random.uniform(key, (B, N, N)) > 0.5
    out2 = ...(q, k, v, mask=mask, num_heads=8)
    expected2 = sdpa_with_mha_and_mask_solution(q, k, v, mask=mask, num_heads=8)
    assert jnp.allclose(out2, expected2)
    print("Q2 is good!")

    # Q3
    out3 = triangle_attn(q, k, v)
    expected3 = triangle_attn_solution(q, k, v)
    assert jnp.allclose(out3, expected3)
    print("Q3 is good!")

    
    return


    # Input: (B, N, N, D)
    in_pw_tensor = jnp.random.rand(B, N, N, D)
    # Output: (B, N, N, D)
    out = attention_forward(x)
    return out

if __name__ == "__main__":
    main()