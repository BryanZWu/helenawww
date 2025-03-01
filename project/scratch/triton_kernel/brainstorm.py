import math
import random
import torch
import torch.nn.functional as F

INV_LOG2 = 1 / math.log(2)

# First, non-triangular attention to test my understanding
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

    # Step 3b: Do manual softmax and make sure it matches
    # Also need to compute logsumexp, the original (not reduced by max for numerical stability)
    # We return log2sumexp.
    def log2sumexp(x):
        x = x * INV_LOG2 # 1.44269504 = 1/log(2)
        max_x = x.max(dim=-1, keepdim=True).values
        x = x - max_x
        stable_sum = torch.sum(torch.exp2(x), dim=-1)
        return torch.log2(stable_sum) + max_x.squeeze(-1)
    logsumexp = log2sumexp(qk_scaled_bias)
    p_manual = torch.exp2(qk_scaled_bias * INV_LOG2 - logsumexp[..., None])
    assert torch.allclose(p, p_manual, atol=1e-5)

    # Step 4: Output
    ref_out = torch.matmul(p, v)

    return ref_out, logsumexp

# Reference forward and 

@torch.no_grad()
def reference_tt_tiled(q, k, v, b, sm_scale, BLOCK_N=64, BLOCK_M=64):
    """
    CPU-computed reference implementation that follows the same tiling strategy as flash attention.
    This matches the Triton kernel's block-based computation pattern for easier debugging.

    Args:
        q: Query tensor [B, H, N, N, D]
        k: Key tensor [B, H, N, N, D]
        v: Value tensor [B, H, N, N, D]
        b: Bias tensor [B, H, N, N]
        sm_scale: Scale factor for attention scores
        BLOCK_N: Block size for tiling (default: 64)
    """
    Z, H, N_CTX, _, D = q.shape
    out = torch.zeros_like(q)
    
    # 1 if we set the output at all--basically a coverage check
    out_set = torch.zeros_like(q)

    # These are filled in in blocks from the inner loop.
    logsumexp_agg = torch.full((Z, H, N_CTX, N_CTX), float('-inf'), device=q.device)  # Max scores

    out_dict = {
        "out": out,
        # "m_i": m_i,
        # "l_i": l_i,
    }

    # Process each batch and head
    for z in range(Z):
        for h in range(H):
            # Process each outer sequence position
            for n1 in range(N_CTX):
                for start_m in range(0, N_CTX, BLOCK_M): # QN2--this is sharded across program_id(0)
                    m_i = torch.full((BLOCK_M,), float('-inf'), device=q.device)
                    l_i = torch.full((BLOCK_M,), 1.0, device=q.device)
                    acc = torch.zeros((BLOCK_M, D), device=q.device)
                    end_m = min(start_m + BLOCK_M, N_CTX)
                    # Process blocks of inner sequence
                    for start_n in range(0, N_CTX, BLOCK_N):  # This loop is equivalent to the inner loop in the triton kernel
                        end_n = min(start_n + BLOCK_N, N_CTX)
                        block_size = end_n - start_n

                        # Get current block of query, key, value
                        q_block = q[z, h, n1, start_m:end_m, :]  # [BLOCK_M, D]
                        k_block = k[z, h, n1, start_n:end_n, :]  # [BLOCK_N, D]
                        v_block = v[z, h, n1, start_n:end_n, :]  # [BLOCK_N, D]
                        b_block = b[z, h, n1, start_n:end_n]  # [BLOCK_N]

                        # Compute attention scores for this block
                        qk = torch.matmul(q_block, k_block.transpose(-2, -1))  # [N_CTX, BLOCK_N]
                        qk_scaled = qk * sm_scale 
                        qk_scaled_bias = qk_scaled + b_block
                        qk_scaled_bias_with_log2 = qk_scaled_bias * 1.44269504 # 1.44269504 = 1/log(2)

                        # Update running max scores
                        m_ij = torch.max(qk_scaled_bias_with_log2, dim=-1)[0]  # [BLOCK_M]
                        m_ij = torch.maximum(m_i, m_ij)  # [BLOCK_M]

                        # Compute attention probabilities
                        qk_scaled_bias_minus_max = qk_scaled_bias_with_log2 - m_ij.unsqueeze(-1)  # Subtract new max for stability
                        p = torch.exp2(qk_scaled_bias_minus_max)

                        # Update sum for normalization
                        l_ij = torch.sum(p, dim=-1)  # [BLOCK_M]
                        alpha = torch.exp2(m_i - m_ij)
                        l_i = l_i * alpha + l_ij
                        acc = acc * alpha[:, None] + p @ v_block

                        # Update max scores for next iteration
                        m_i = m_ij

                    # Final normalization
                    acc = acc / l_i[:, None]
                    logsumexp_agg[z, h, n1, start_m:end_m] = m_i + torch.log2(l_i)
                    out[z, h, n1, start_m:end_m, :] = acc

    return out, logsumexp_agg


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
        # da = (diag(p) - p.T) @ dP
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

def manual_tile_backward(d_out, Q, K, V, B_pw, O, sm_scale, logsumexp_agg, BLOCK_QL2=64, BLOCK_KL2=64):
    """
    Manual implementation of tile backward.
    Note that we don't use P and instead rematerialize it to 
    avoid global memory access.

    In practice the backward pass is _two_ separate kernels.
    """
    B, H, QL1, QL2, D = Q.shape
    B, H, KL1, KL2, D = K.shape
    # Kernel 1: preprocessing to compute elementwise multiple of 
    # O and d_out. This is a no-op from the other kernel for now
    OdO_sum = (O * d_out).sum(dim=-1)
    assert OdO_sum.shape == (B, H, QL1, QL2)

    BLOCK_QL2 = 32
    BLOCK_KL2 = 32

    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    dQ = torch.zeros_like(Q)
    dB_pw = torch.zeros_like(B_pw)

    # Kernel 2: Here we want to tile as needed
    # First 3 for loops are handeld by program id tiling
    for b in range(B):
        for h in range(H):
            for l1 in range(0, QL1):
                for n in range(0, KL2, BLOCK_KL2):
                    dv_n = torch.zeros((BLOCK_KL2, D))
                    dk_n = torch.zeros((BLOCK_KL2, D))
                    k = K[b, h, l1, n:n+BLOCK_KL2]
                    v = V[b, h, l1, n:n+BLOCK_KL2]
                    for m in range(0, QL1, BLOCK_QL2):
                        # Rematerialize A and P
                        q = Q[b, h, l1, m:m+BLOCK_QL2]
                        l = logsumexp_agg[b, h, l1, m:m+BLOCK_QL2]
                        assert l.shape == (BLOCK_QL2,)
                        assert q.shape == (BLOCK_QL2, D)
                        assert k.shape == (BLOCK_KL2, D)
                        assert v.shape == (BLOCK_KL2, D)

                        a_prenorm_prebias = q @ k.mT
                        a_prebias = a_prenorm_prebias * sm_scale
                        # No l1 shape
                        a = a_prebias + B_pw[b, h, m:m+BLOCK_QL2, n:n+BLOCK_KL2]
                        # L is the logsumexp denominator
                        p = torch.exp(a - l)
                        d_out_tile = d_out[b, h, l1, m:m+BLOCK_QL2]
                        dv = p @ d_out_tile
                        dp = d_out_tile @ v.mT
                        # Inverse softmax
                        da = p * (dp - OdO_sum[b, h, l1, m:m+BLOCK_QL2])
                        dQ[b, h, l1, m:m+BLOCK_QL2] += da @ k
                        dk = da.mT @ q # transpose here?
                        dv_n += dv
                        dk_n += dk
                    dV[b, h, l1, n:n+BLOCK_KL2] = dv_n
                    dK[b, h, l1, n:n+BLOCK_KL2] = dk_n
    return dQ, dK, dV, dB_pw

def test_manual_tile_backward():
    # Forward pass and backward pass, followed by forward tiled pass 
    # and backward tiled pass.
    # Then, check that the tiled backward pass matches the manual one.
    B, H, L, D = 2, 4, 128, 64
    Q = torch.randn(B, H, L, L, D, requires_grad=True)
    K = torch.randn(B, H, L, L, D, requires_grad=True)
    V = torch.randn(B, H, L, L, D, requires_grad=True)
    B_pw = torch.randn(B, H, L, L, requires_grad=True)
    sm_scale = D ** -0.5

    # Torch autograd forward/backward
    out_torch, logsumexp_torch = reference_tt_attn(Q, K, V, B_pw, sm_scale)
    loss = out_torch.sum()
    loss.backward()
    dQ_autograd = Q.grad
    dK_autograd = K.grad
    dV_autograd = V.grad
    dB_pw_autograd = B_pw.grad

    # Tiled manual forward pass
    out, logsumexp_agg = reference_tt_tiled(Q, K, V, B_pw, sm_scale)
    assert torch.allclose(logsumexp_agg, logsumexp_torch, atol=1e-5)
    assert torch.allclose(out, out_torch, atol=1e-5)

    # The grad of sum is just 1
    d_out = torch.ones_like(out)

    # Tiled manual backward pass
    dQ, dK, dV, dB_pw = manual_tile_backward(d_out, Q, K, V, B_pw, out, sm_scale, logsumexp_agg)
    assert torch.allclose(dQ, dQ_autograd, atol=1e-5)
    assert torch.allclose(dK, dK_autograd, atol=1e-5)
    assert torch.allclose(dV, dV_autograd, atol=1e-5)
    assert torch.allclose(dB_pw, dB_pw_autograd, atol=1e-5)

if __name__ == "__main__":
    # test_manual_attention_gradients()
    # test_manual_triangular_attention_gradients()
    torch.manual_seed(0)
    random.seed(0)
    test_manual_tile_backward()
    