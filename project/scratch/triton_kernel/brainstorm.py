import torch
import torch.nn.functional as F

def reference_tt_attn(q, k, v, b, sm_scale):
    """
    Reference implementation of attention for correctness.

    If a debugging_dict is provided, we will check intermediate values against the debugging_dict.
    """
    Z, H, L = q.shape[:3]
    # Step 1: QK^T. 
    qk = torch.matmul(q, k.transpose(-2, -1))

    # Step 2: Scale and add bias--attn scaling happens before bias addition
    qk_scaled = qk * sm_scale 
    qk_scaled_bias = qk_scaled + b.view(Z, H, 1, L, L)

    # Step 3a: Softmax (builtin)
    p = torch.softmax(qk_scaled_bias.float(), dim=-1).to(v.dtype)

    # Step 4: Output
    ref_out = torch.matmul(p, v)
    return ref_out

def reference_attn(q, k, v, sm_scale):
    """
    Reference implementation of general attention for correctness.
    """
    attn_matrix = torch.matmul(q, k.transpose(-2, -1))
    attn_matrix = torch.softmax(attn_matrix * sm_scale, dim=-1)
    out = torch.matmul(attn_matrix, v)
    return out

def manual_attention_gradients(d_out, q, k, v, p, sm_scale):
    """
    Manual implementation of attention gradients.
    """
    B, H, L, D = q.shape
    assert q.shape == k.shape == v.shape
    assert d_out.shape == (B, H, L, D)
    assert p.shape == (B, H, L, L)
    dV = torch.matmul(p.transpose(-2, -1), d_out)
    assert dV.shape == (B, H, L, D)
    dP = torch.matmul(d_out, v.transpose(-2, -1))
    assert dP.shape == (B, H, L, L)
    def d_softmax(p, dP):
        return dP * p * (1 - p)
    d_attn = d_softmax(p, dP)
    assert d_attn.shape == (B, H, L, L)
    dQ = torch.matmul(d_attn, k)
    assert dQ.shape == (B, H, L, D)
    dK = torch.matmul(d_attn.transpose(-2, -1), q)
    assert dK.shape == (B, H, L, D)
    return dQ, dK, dV

def test_manual_attention_gradients():
    
