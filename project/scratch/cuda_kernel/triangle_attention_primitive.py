from functools import partial
import jax
from jax import core
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.lib import xla_client
import numpy as np
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo

# tutorial had:
# for _name, _value in gpu_ops.get_rms_norm_registrations().items():

for _name, _value in ...: 
    # Load the shared library
    xla_client.register_custom_call_target(
        _name,
        _value,
        platform="cuda",
    )

# Define the primitive
triangle_attention_p = core.Primitive("triangle_attention")
triangle_attention_p.multiple_results = False  # Single output tensor

# Function provided to the user
def triangle_attention(q, k, v, mask, *, from_starting_node=True):
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
    return triangle_attention_p.bind(
        q, k, v, mask,
        from_starting_node=from_starting_node
    )

# Define abstract evaluation (shape inference)
def triangle_attention_abstract_eval(q, k, v, mask, *, from_starting_node):
    assert q.shape == k.shape == v.shape
    assert len(q.shape) == 4
    assert len(mask.shape) == 4
    return core.ShapedArray(q.shape, q.dtype)

triangle_attention_p.def_abstract_eval(triangle_attention_abstract_eval)

# Define the MLIR lowering
def triangle_attention_lowering(ctx, q, k, v, mask, *, from_starting_node):
    # Get input types and shapes
    q_type = ir.RankedTensorType(q.type)
    q_shape = q_type.shape
    dtype = q_type.element_type
    
    # Create descriptor
    descriptor = np.array([
        q_shape[0],  # batch_size
        q_shape[1],  # seq_length
        q_shape[3] // 64,  # num_heads
        64,  # head_dim
        1 if from_starting_node else 0,  # from_starting_node
    ], dtype=np.int32).tobytes()
    
    # Output type
    out_type = ir.RankedTensorType.get(q_shape, dtype)
    
    # Create custom call
    return [
        mhlo.CustomCallOp(
            [out_type],
            [q, k, v, mask],
            call_target_name = ir.StringAttr.get("triangle_attention_cuda"),
            has_side_effect = ir.BoolAttr.get(False),
            backend_config = ir.StringAttr.get(descriptor),
            api_version = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 1),
            called_computations = ir.ArrayAttr.get([]),
            operand_layouts = ir.ArrayAttr.get([]),
            result_layouts = ir.ArrayAttr.get([]),
        ).result
    ]

mlir.register_lowering(
    triangle_attention_p,
    triangle_attention_lowering,
    platform="cuda"
) 