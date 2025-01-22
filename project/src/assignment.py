# torch equivalent:
from jax import numpy as jnp
import jax
import math
from reference import scaled_dot_product, sdpa_with_mha_and_mask, triangle_attn

B, N, D = 2, 256, 128
print("JAX devices:", jax.devices())

@jax.jit
def vanilla_attention_forward(q, k, v):
    # Implement the vanilla attention forward pass: 
    # attn_out = Q @ K^T / sqrt(D)
    ...
    D = q.shape[-1] # its fine just push the code to edit locally
    return q @ k.T / math.sqrt(D)

def main(): 
    # First, make sure that the vanilla attention forward pass works
    # is it this command pip install -U "jax[cuda12]" torch? ohkay

    B, N, N, D = 2, 256, 256, 128

    key = jax.random.PRNGKey(0)
    q = jax.random.uniform(key, (B, N, D))
    k = jax.random.uniform(key, (B, N, D))
    v = jax.random.uniform(key, (B, N, D))
    out1 = scaled_dot_product(q, k, v)
    expected1 = vanilla_attention_forward(q, k, v)
    assert jnp.allclose(out1, expected1)
    print("Q1 is good!")

    # Q2
    mask = jax.random.uniform(key, (B, N, N)) > 0.5
    out2 = sdpa_with_mha_and_mask(q, k, v, mask=mask, num_heads=8)
    expected2 = vanilla_attention_forward(q, k, v)
    assert jnp.allclose(out2, expected2)
    print("Q2 is good!")

    # Q3
    out3 = triangle_attn(q, k, v)
    expected3 = vanilla_attention_forward(q, k, v)
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