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

def manual_attention_gradients(d_out, Q, K, V, P, sm_scale):
    """
    Manual implementation of attention gradients.
    """
    B, H, L, D = Q.shape
    assert Q.shape == K.shape == V.shape
    assert d_out.shape == (B, H, L, D)
    assert P.shape == (B, H, L, L)
    dV = P.mT @ d_out
    assert dV.shape == (B, H, L, D)
    dP = d_out @ V.mT
    assert dP.shape == (B, H, L, L)
    def d_softmax(P, dP):
        dA = torch.zeros_like(dP)
        # da = p(I-p.T)dP
        for j in range(L):
            p = P[:, :, j]
            diag_p = torch.diag_embed(p)
            assert diag_p.shape == (B, H, L, L)
            p_outer_p = p[:, :, None, :] * p[:, :, :, None]
            assert p_outer_p.shape == (B, H, L, L)
            dA[:, :, j, :] = ((diag_p - p_outer_p) @ dP[:, :, j, :, None]).squeeze(-1)
        return dA
    d_scaled_A = d_softmax(P, dP)
    dA = d_scaled_A * sm_scale
    assert dA.shape == (B, H, L, L)
    dQ = dA @ K
    assert dQ.shape == (B, H, L, D)
    dK = dA.mT @ Q
    assert dK.shape == (B, H, L, D)
    return dQ, dK, dV, dP, dA

def test_manual_attention_gradients():
    B, H, L, D = 2, 4, 8, 16
    Q = torch.randn(B, H, L, D, requires_grad=True)
    K = torch.randn(B, H, L, D, requires_grad=True)
    V = torch.randn(B, H, L, D, requires_grad=True)
    sm_scale = D ** -0.5

    # First, run through vanilla attention
    A = Q @ K.transpose(-2, -1)
    A.retain_grad()
    P = torch.softmax(A * sm_scale, dim=-1)
    P.retain_grad()
    out = P @ V
    out.retain_grad()

    loss = out.sum()

    # Autograd gradients
    loss.backward()
    d_out = out.grad
    dQ_autograd = Q.grad
    dK_autograd = K.grad
    dV_autograd = V.grad
    dP_autograd = P.grad
    dA_autograd = A.grad

    # Now, run through manual gradients
    dQ, dK, dV, dP, dA = manual_attention_gradients(d_out, Q, K, V, P, sm_scale)
    assert torch.allclose(dP, dP_autograd, atol=1e-5)
    assert torch.allclose(dA, dA_autograd, atol=1e-5)
    assert torch.allclose(dQ, dQ_autograd, atol=1e-5)
    assert torch.allclose(dK, dK_autograd, atol=1e-5)
    assert torch.allclose(dV, dV_autograd, atol=1e-5)

def manual_triangular_attention_gradients(d_out, Q, K, V, B_pw, P, sm_scale):
    """
    Manual implementation of triangular attention gradients.
    """
    B, H, L, L, D = Q.shape
    assert Q.shape == K.shape == V.shape
    assert d_out.shape == (B, H, L, L, D)
    assert P.shape == (B, H, L, L, L)
    dV = P.mT @ d_out
    assert dV.shape == (B, H, L, L, D)
    dP = d_out @ V.mT
    assert dP.shape == (B, H, L, L, L)
    def d_softmax(P, dP):
        dA = torch.zeros_like(dP)
        # da = p(I-p.T)dP
        for j in range(L):
            p = P[:, :, :, j]
            diag_p = torch.diag_embed(p)
            assert diag_p.shape == (B, H, L, L, L)
            p_outer_p = p[:, :, :, :, None] * p[:, :, :, None, :]
            assert p_outer_p.shape == (B, H, L, L, L)
            dA[:, :, :, j, :] = ((diag_p - p_outer_p) @ dP[:, :, :, j, :, None]).squeeze(-1)
        return dA
    d_scaled_A = d_softmax(P, dP) # Shape (B, H, L, L, L)
    assert d_scaled_A.shape == (B, H, L, L, L)
    dB_pw = d_scaled_A.sum(dim=2)
    dA = d_scaled_A * sm_scale
    assert dA.shape == (B, H, L, L, L)
    dQ = dA @ K
    assert dQ.shape == (B, H, L, L, D)
    dK = dA.mT @ Q
    assert dK.shape == (B, H, L, L, D)
    return dQ, dK, dV, dB_pw, dP, dA

def test_manual_triangular_attention_gradients():
    B, H, L, D = 2, 4, 8, 16
    Q = torch.randn(B, H, L, L, D, requires_grad=True)
    K = torch.randn(B, H, L, L, D, requires_grad=True)
    V = torch.randn(B, H, L, L, D, requires_grad=True)
    B_pw = torch.randn(B, H, L, L, requires_grad=True)
    sm_scale = D ** -0.5

    # First, run through vanilla attention
    A = Q @ K.mT
    A.retain_grad()
    P = torch.softmax(A * sm_scale + B_pw.view(B, H, 1, L, L), dim=-1)
    P.retain_grad()
    out = P @ V
    out.retain_grad()

    loss = out.sum()

    # Autograd gradients
    loss.backward()
    d_out = out.grad
    dQ_autograd = Q.grad
    dK_autograd = K.grad
    dV_autograd = V.grad
    dP_autograd = P.grad
    dA_autograd = A.grad
    dB_pw_autograd = B_pw.grad

    # Now, run through manual gradients
    dQ, dK, dV, dB_pw, dP, dA = manual_triangular_attention_gradients(d_out, Q, K, V, B_pw, P, sm_scale)
    assert torch.allclose(dP, dP_autograd, atol=1e-5)
    assert torch.allclose(dA, dA_autograd, atol=1e-5)
    assert torch.allclose(dQ, dQ_autograd, atol=1e-5)
    assert torch.allclose(dK, dK_autograd, atol=1e-5)
    assert torch.allclose(dV, dV_autograd, atol=1e-5)
    assert torch.allclose(dB_pw, dB_pw_autograd, atol=1e-5)

if __name__ == "__main__":
    test_manual_attention_gradients()
    test_manual_triangular_attention_gradients()
    
    