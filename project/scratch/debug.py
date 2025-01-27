# torch equivalent:
from jax import numpy as jnp
import jax
import torch
import math

B, N, D = 2, 256, 128

@jax.jit
def vanilla_attention_forward(q, k, v, num_heads=8, head_dim=64):
    """
    Pure JAX implementation of attention.

    Q: query K: key V: value
    Creates at attention mask, such 

    A cat is a Dog.
    what are qkv in here
    A cat is a Dog. # N by N
  A | 9   |  |  |
 cat| |   |  |  |
is  | |   |  |  |
 a  | |   |  0  9
dog | |   |  |  |

    idea:
    access items from a dictionary or map?
    {happy: 9, sad: 0. Okay: 5} <- dictionary of "values" that indicate how happy something is
    question: how can we just access this?
    "I want to know if there is antoher word in this sentence that is happy" -> 
    (you can check the values of 9, 0, and 5, and then select "happy" since it's closest) 
    closets to what I'm looking for which is "positive"
    
    problem is that just picking the max -> non differentiable. Not a continuous function! 
    instead -> softmax. (basically a continuous version that appraches the max)

    softmax(x_i) = exp(x_i) / sum(exp(x_i))
    x_i is the "attention score", sum of softmax(x_i) is 1. Natural interpretation of "probability".
    apply softmax to attention scores, now you have a matrix of probabilities.

    A cat is a Dog. # N by N
  A | .93   |  |  |
 cat| |   |  |  |
is  | |   |  |  |
 a  | |   |  .001  .93 (they sum to 1)
dog | |   |  |  |

    Once you have the map, you can use it to get the info that you're interested in. 

    Stepping back now to the whole QKV thing.

    Q: query K: key V: value

    "I want a word that's positive, and I want to know if that word is describing a dog"
    Q: "I want a word that's positive"
    K: "The 'postivity' of each word"
    V: "what the word describes"

    combine Q and K to dictate how much attention to pay, and then once you know how much attention 
    to pay, you would fetch the value of what the word describes. 
    (downstream you would do something with what the word is describing)
    
    How do we combine Q and K?
    Take dot product -> (D) dot (D) -> float/scalar. Do this for each word, to each other word.
    (N, D) dot (N, D) -> (N, N) (softmaxed out -> they sum to 1)

    Afterwards just apply to values.

    attention map is softmax(QK^T) dot V
    Q: B, N, D
    K: B, N, D (transpoed becomes B, D, N) 
    QK^T: B, N, N (how much each word pays to each other word)
    softmax(QK^T): same thing, now sums to 1, interpret as weight/probability
    softmax(QK^T) dot V: B, N, D (actual thing that you "looked up")
    hm is it possible you split out some work to me like define some func to implement lol
    """

    # attention map shape (B, N, N)
    attention_matmul = q @ jnp.swapaxes(k, -2, -1)
    attention_matmul = attention_matmul / jnp.sqrt(D)
    attention_softmax = jax.nn.softmax(attention_matmul, axis=-1)
    return attention_softmax @ v

    
    return _attention_fn(q, k, v)

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = jax.nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values

def main(): 
    # First, make sure that the vanilla attention forward pass works
    # is it this command pip install -U "jax[cuda12]" torch? ohkay

    B, N, N, D = 2, 256, 256, 128

    key = jax.random.PRNGKey(0)
    q = jax.random.uniform(key, (B, N, D))
    k = jax.random.uniform(key, (B, N, D))
    v = jax.random.uniform(key, (B, N, D))
    out = scaled_dot_product(q, k, v)
    out2 = vanilla_attention_forward(q, k, v)
    assert jnp.allclose(out, out2)
    print(out.shape)
    return 

    # Input: (B, N, N, D)
    in_pw_tensor = jnp.random.rand(B, N, N, D)
    # Output: (B, N, N, D)
    out = attention_forward(x)
    return out

if __name__ == "__main__":
    main()