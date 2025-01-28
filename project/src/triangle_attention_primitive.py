from functools import partial
import jax
from jax import core
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.lib import xla_client
import numpy as np
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo

"""
TLDR: 
Jax operates on "primitives" which everything else builds on. When you make a 
custom kernel you usually register as a primitive. 

When jax does an operation using jit, it does it at least twice:
- once on compile time 
- once every time the function is run.

We need to define both of these separately. Thus, the steps are as follows:

1. Register the custom call target
    - Allows us to use the compiled kernel and call it later 
      in python via custom_call(...)
2. Define the primitive, including the abstract evaluation and 
    the MLIR lowering (aka the actual kernel)
    - core.Primitive() creates the primitive object
    - def_abstract_eval() defines the abstract evaluation--shapes/types
    - def_lowering() is the link betwene the python and the GPU. Uses
      custom_call(...) to call the kernel and returns it. MOST IMPORTANT PART!
3. Register all of the above with the primitive

(Please don't ask me what MLIR is because I don't know)
"""



# 1. Register the custom call target
xla_client.register_custom_call_target(
    "triangle_attention_cuda",
    xla_client.get_triangle_attention_cuda_library("cuda/libtriangle_attention.so")
)

# 2. Define the primitive
triangle_attention_fwd_p = core.Primitive("triangle_attention_fwd")
triangle_attention_fwd_p.multiple_results = False

triangle_attention_bwd_p = core.Primitive("triangle_attention_bwd")
triangle_attention_bwd_p.multiple_results = False

def triangle_attention_fwd(q, k, v, mask, *, from_starting_node=True):
    """Triangle attention primitive.
    
    Args:
        q: Query tensor of shape [batch_size, seq_len, seq_len, hidden_dim]
        k: Key tensor of shape [batch_size, seq_len, seq_len, hidden_dim]
        v: Value tensor of shape [batch_size, seq_len, seq_len, hidden_dim]
        mask: Attention mask of shape [batch_size * num_heads, seq_len, seq_len, seq_len]
        from_starting_node: Whether attention flows from starting node
    
    Returns:
        Output tensor of shape [batch_size, seq_len, seq_len, hidden_dim]
    """
    return triangle_attention_fwd_p.bind(
        q, k, v, mask,
        from_starting_node=from_starting_node
    )

def triangle_attention_bwd():
    raise NotImplementedError("Triangle attention backward not implemented")
    return triangle_attention_bwd_p.bind(
        q, k, v, mask,
        from_starting_node=from_starting_node
    )

# 3. Define abstract evaluation (shape inference)
def triangle_attention_fwd_abstract_eval(q, k, v, mask, from_starting_node):
    # For now assumes that q, k, v are all the same shape, since pw stuff
    assert q.shape == k.shape == v.shape == mask.shape
    B, L, L, H = q.shape
    # Shape checking logic
    return core.ShapedArray((B, L, L, H), q.dtype)

def triangle_attention_bwd_abstract_eval():
    raise NotImplementedError("Triangle attention backward abstract eval not implemented")
    # For now assumes that q, k, v are all the same shape, since pw stuff
    assert q.shape == k.shape == v.shape == mask.shape
    B, L, L, H = q.shape
    # Shape checking logic
    return core.ShapedArray((B, L, L, H), q.dtype)

triangle_attention_fwd_p.def_abstract_eval(triangle_attention_fwd_abstract_eval)
# triangle_attention_bwd_p.def_abstract_eval(triangle_attention_bwd_abstract_eval)

# 4. Define MLIR lowering
def triangle_attention_fwd_lowering(ctx, *args, **kwargs):
    # MLIR lowering logic
    return [mhlo.CustomCallOp(...)]

mlir.register_lowering(
    my_primitive,
    my_primitive_lowering,
    platform="cuda"
)