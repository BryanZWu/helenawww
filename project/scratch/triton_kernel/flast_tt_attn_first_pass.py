'''
Code adapted from the triton flash-attn-2.0 tutorial:
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

Modifications: 
- In AF2 triangular attention, an additional bias is applied to the 
  attention scores: "The decision whether edge ij will receive an
  update from edge ik is not only determined by their
  query-key similarity (as in standard attention [95]),
  but also modulated by the information bjk derived from
  the third edge jk of this triangle"
- Triangle attention input is pw for all QKV: (B, H, N, N, D)
'''

import pytest
import torch
# import triton.tools.experimental_descriptor

import triton
import triton.language as tl

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Helper function to check if we're using HIP (AMD) backend
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

# Helper function to check if we're using CUDA backend
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

print("TMA benchmarks will be running without grid constant TMA descriptor.")


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr, B_block_ptr, #
                    start_qn2, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    offs_qn2: tl.constexpr, offs_kn2: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    """
    Inner kernel for attention forward pass computation
    Computes attention scores and updates accumulator for a block of the attention matrix
    """
    lo, hi = 0, N_CTX

    # Add debug prints for first thread
    if ((tl.program_id(0) == 0) and (tl.program_id(1) == 0)) and (tl.program_id(2) == 0):
        tl.debug_barrier()
        tl.device_print("start_qn2:", start_qn2)
        tl.device_print("BLOCK_M:", BLOCK_M)
        tl.device_print("BLOCK_N:", BLOCK_N)
        tl.device_print("q shape:", q.shape[0], q.shape[1])

    # Advance block pointers to correct starting positions
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    B_block_ptr = tl.advance(B_block_ptr, (0, lo))

    # Process blocks of K and V to compute attention
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        if ((tl.program_id(0) == 0) and (tl.program_id(1) == 0)) and (tl.program_id(2) == 0):
            tl.device_print("Processing block start_n:", start_n)

        # Load and print K block shape
        k = tl.load(K_block_ptr)
        if ((tl.program_id(0) == 0) and (tl.program_id(1) == 0)) and (tl.program_id(2) == 0):
            tl.device_print("k shape:", k.shape[0], k.shape[1])

        qk = tl.dot(q, k)

        # Print QK shape and some values
        if ((tl.program_id(0) == 0) and (tl.program_id(1) == 0)) and (tl.program_id(2) == 0):
            tl.device_print("qk shape:", qk.shape[0], qk.shape[1])
            # tl.device_print("qk[0,0]:", qk[0,0])

        # Apply attention bias 
        b = tl.load(B_block_ptr)
        qk = qk * qk_scale + b
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None] # Subtract max value for numerical stability

        # Compute softmax values
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # Update accumulators
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Load V and compute attention output
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)  # Convert to FP8 if needed
        else:
            p = p.to(tl.float16)  # Otherwise use FP16
        acc = tl.dot(p, v, acc)  # Update accumulator with attention values

        # Update tracking variables
        m_i = m_ij

        # Advance block pointers
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i


# @triton.jit
# def _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
#                         desc_k, desc_v,  #
#                         offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
#                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
#                         STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
#                         N_CTX: tl.constexpr):
#     """
#     TMA (Tensor Memory Access) version of the attention forward pass kernel
#     Uses TMA descriptors for more efficient memory access patterns
#     Similar to _attn_fwd_inner but optimized for TMA operations
#     """
#     # range of values handled by this stage
#     if STAGE == 1:
#         lo, hi = 0, start_m * BLOCK_M
#     elif STAGE == 2:
#         lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
#         lo = tl.multiple_of(lo, BLOCK_M)
#     # causal = False
#     else:
#         lo, hi = 0, N_CTX
#     offsetkv_y = offset_y + lo
#     # loop over k, v and update accumulator
#     for start_n in range(lo, hi, BLOCK_N):
#         start_n = tl.multiple_of(start_n, BLOCK_N)
#         # -- compute qk ----
#         k = tl._experimental_descriptor_load(desc_k, [offsetkv_y, 0], [BLOCK_N, HEAD_DIM], dtype).T
#         qk = tl.dot(q, k)
#         if STAGE == 2:
#             mask = offs_m[:, None] >= (start_n + offs_n[None, :])
#             qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
#             m_ij = tl.maximum(m_i, tl.max(qk, 1))
#             qk -= m_ij[:, None]
#         else:
#             m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
#             qk = qk * qk_scale - m_ij[:, None]
#         p = tl.math.exp2(qk)
#         l_ij = tl.sum(p, 1)
#         # -- update m_i and l_i
#         alpha = tl.math.exp2(m_i - m_ij)
#         l_i = l_i * alpha + l_ij
#         # -- update output accumulator --
#         acc = acc * alpha[:, None]
#         # update acc
#         v = tl._experimental_descriptor_load(desc_v, [offsetkv_y, 0], [BLOCK_N, HEAD_DIM], dtype)
#         p = p.to(dtype)
#         # note that this non transposed v for FP8 is only supported on Blackwell
#         acc = tl.dot(p, v, acc)
#         # update m_i and l_i
#         m_i = m_ij
#         offsetkv_y += BLOCK_N
#     return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["n", "d"])
@triton.jit
def _attn_fwd(Q, K, V, B, sm_scale, M, Out,  # sm_scale: scaling factor for softmax, M: max values, Out: output tensor
              stride_qb, stride_qh, stride_qn1, stride_qn2, stride_qd,  # Strides for Q tensor: batch, head, seq_len, seq_len, head_dim
              stride_kb, stride_kh, stride_kn1, stride_kn2, stride_kd,  # Strides for K tensor
              stride_vb, stride_vh, stride_vn1, stride_vn2, stride_vd,  # Strides for V tensor
              stride_ob, stride_oh, stride_on1, stride_on2, stride_od,  # Strides for Output tensor
              stride_bb, stride_bh, stride_bn1, stride_bn2,  # Strides for B tensor
              b, h, n,  # b=batch_size, h=num_heads, n=sequence_length
              d: tl.constexpr,  # Size of each attention head
              BLOCK_M: tl.constexpr,  # Block size for sequence dimension
              BLOCK_N: tl.constexpr,  # Block size for attention computation
              ):
    """
    Attention forward pass kernel: 
    Q: [b, h, n, n, d]
    K: [b, h, n, n, d]
    V: [b, h, n, n, d]
    B: [b, h, n, n]


    Naive implementation:
    1. attn_logits = Q @ K.transpose(-2, -1) # [B, H, N, N, N]
    2. attn_logits = attn_logits / math.sqrt(head_dim)
    3. attn_logits = attn_logits + bjk # [B, H, N, N, N]
    4. attn_score = softmax(attn_logits, axis=-1) # [B, H, N, N, N]
    5. attn_out = attn_score @ V # [B, H, N, N, D]

    Of these, step 3 is not handled by out-of-the-box flash attention.

    Note on nomenclature:
    - b: batch dimension
    - h: head dimension
    - n1: outer sequence dimension (where we tile/aggregate)
    - n2: inner sequence dimension (where we compute attention)
    - d: head dimension

    - m: row dim of attn matrix (n2 for Q)
    - n: col dim of attn matrix (n2 for K/V)

    (n1/n2 is not the same as N!)
    """
    # Add debug prints for first thread
    if ((tl.program_id(0) == 0) and (tl.program_id(1) == 0)) and (tl.program_id(2) == 0):
        tl.debug_barrier()
        tl.device_print("Grid dims:", tl.num_programs(0), tl.num_programs(1), tl.num_programs(2))
        tl.device_print("Block sizes: BLOCK_M=", BLOCK_M)
        tl.device_print("BLOCK_N=", BLOCK_N)
        tl.device_print("Input dims: b=", b)
        tl.device_print("h=", h)
        tl.device_print("n=", n)
        tl.device_print("d=", d)

    # Ensure block size doesn't exceed head dimension
    tl.static_assert(BLOCK_N <= d, "Block size (BLOCK_N) must not exceed head dimension (d)")

    # Grid dimension explanation:
    # - program_id(0): Indexes along inner sequence length (N_CTX/BLOCK_M blocks)
    #      the N along which we aggregate q and k
    # - program_id(1): Indexes along outer sequence length (N_CTX blocks)
    # - program_id(2): Indexes along batch*heads dimension (Z*H blocks)
    start_qn2 = tl.program_id(0)  # Inner sequence dimension (where we tile/aggregate)
    off_n1 = tl.program_id(1) # Outer sequence dimension
    off_bh = tl.program_id(2)   # Combined batch and head index

    # Convert combined batch-head index into separate indices
    off_b = off_bh // h         # Batch index
    off_h = off_bh % h          # Head index

    # Calculate base offset for current batch and head, and also first dim of N
    # (which is effectively just another dim to batch along for now)
    # Note that we assume similar strides for Q, K, V, as the original
    # triton flash attention kernel does.
    qvk_offset = (
        off_b.to(tl.int64) * stride_qb +
        off_h.to(tl.int64) * stride_qh +
        off_n1.to(tl.int64) * stride_qn1
    )
    b_offset = (
        off_b.to(tl.int64) * stride_bb +
        off_h.to(tl.int64) * stride_bh
    )

    # Flash attention 2 slices Q instead of KV slicing, so that the output
    # is written by one warp/SM. 

    # Create block pointers for efficient memory access:
    # Q: Current block of queries [BLOCK_M, HEAD_DIM]
    # K: Current block of keys [HEAD_DIM, BLOCK_N]
    # V: Current block of values [BLOCK_N, HEAD_DIM]
    # O: Output block [BLOCK_M, HEAD_DIM]
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(n, d),
        strides=(stride_qn2, stride_qd),
        offsets=(start_qn2 * BLOCK_M, 0), # Tile Q across blocks
        block_shape=(BLOCK_M, d),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    # TODO: will need to understand what's going on with these block ptrs
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(n, d),
        strides=(stride_vn2, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, d),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(d, n),
        strides=(stride_kn2, stride_kd),
        offsets=(0, 0),
        block_shape=(d, BLOCK_N),
        order=(0, 1),
    )
    B_block_ptr = tl.make_block_ptr(
        base=B + b_offset,
        shape=(n, n),
        strides=(stride_bn1, stride_bn2),
        offsets=(start_qn2 * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(n, d),
        strides=(stride_on2, stride_od),
        offsets=(start_qn2 * BLOCK_M, 0),
        block_shape=(BLOCK_M, d),
        order=(1, 0),
    )
    # Initialize per-block data structures:
    # Sequence offsets within the block: N2 dim of Q
    offs_qn2 = start_qn2 * BLOCK_M + tl.arange(0, BLOCK_M)  
    # Attention target offsets within the block: N2 dim of KV
    offs_kn2 = tl.arange(0, BLOCK_N)

    # Initialize tracking variables for softmax computation
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Max values for stability
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0          # Sum for normalization
    acc = tl.zeros([BLOCK_M, d], dtype=tl.float32)      # Accumulated attention outputs

    # softmax scaling factor--since we use log2, we need to scale by 1/log(2)
    qk_scale = sm_scale * 1.44269504  # 1.44269504 = 1/log(2)

    # Load query block - remains in SRAM throughout computation
    q = tl.load(Q_block_ptr)

    acc, l_i, m_i = _attn_fwd_inner(
        acc, l_i, m_i, q,
        K_block_ptr, V_block_ptr, B_block_ptr, #
        start_qn2, qk_scale,  #
        BLOCK_M, d, BLOCK_N,  #
        offs_qn2, offs_kn2,
        n, V.dtype.element_ty == tl.float8e5  #
        )

    # Final processing:
    # 1. Apply softmax normalization
    # 2. Store softmax scaling factors (m_i) for backward pass
    # 3. Store final output values
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_bh * n + offs_qn2
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs_tma = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in [2, 3, 4, 6]\
    for w in [4, 8]\
]



class _attention(torch.autograd.Function):
    """
    Custom PyTorch autograd function for flash attention implementation
    Implements efficient forward and backward passes for the attention mechanism
    """

    @staticmethod
    def forward(ctx, q, k, v, b, sm_scale):
        """
        Forward pass of flash attention
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            causal: Whether to use causal attention
            sm_scale: Softmax scaling factor
        """
        # Verify shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]  # V may be transposed for float8
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        # Initialize output tensor
        # o = torch.empty_like(q)
        o = torch.zeros_like(q)


        # Setup kernel arguments
        extra_kern_args = {}

        # Special tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        # Initialize scaling factor storage
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        def grid(args):
            return (triton.cdiv(q.shape[-2], args["BLOCK_M"]), q.shape[-3], q.shape[0] * q.shape[1])

        ctx.grid = grid

        # Launch regular kernel
        _attn_fwd[grid](
            q, k, v, b, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3), k.stride(4),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3), v.stride(4),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3), o.stride(4),
            b.stride(0), b.stride(1), b.stride(2), b.stride(3),
            b=q.shape[0], h=q.shape[1], n=q.shape[2],
            d=HEAD_DIM_K,
            **extra_kern_args)

        # Save tensors needed for backward pass
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        return o

attention = _attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 32)])
def test_op(Z, H, N_CTX, HEAD_DIM, dtype=torch.float16):
    """
    Test function for attention implementation
    Compares Triton implementation with PyTorch reference implementation
    """
    # Set random seed for reproducibility
    torch.manual_seed(20)

    # Initialize input tensors
    q = (torch.empty((Z, H, N_CTX, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    dout = torch.randn_like(q)

    # Reference implementation using PyTorch
    b = torch.ones((Z, H, N_CTX, N_CTX), device=DEVICE)
    p = torch.matmul(q, k.transpose(3, 4)) * sm_scale  # Compute attention scores
    p = p + b.view(Z, H, 1, N_CTX, N_CTX)
    p = torch.softmax(p.float(), dim=-1).half()  # Apply softmax
    ref_out = torch.matmul(p, v)  # Compute attention output

    # Compute reference gradients
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    # Triton implementation
    tri_out = attention(q, k, v, b, sm_scale).half()

    # Compare results--forward pass
    # Updated debug: only 2 of the 256 elements of N1 is written to. 
    # for (B, H, N1, N, C), the first 2 of N1 are written to. 
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)

    backwards_pass_implemented = False
    if backwards_pass_implemented:
        tri_out.backward(dout)
        tri_dv, v.grad = v.grad.clone(), None
        tri_dk, k.grad = k.grad.clone(), None
        tri_dq, q.grad = q.grad.clone(), None

        rtol = 0.0
        # Handle special case for AMD MI200 GPU
        if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
            rtol = 1e-2  # Adjust tolerance for known hardware limitation

        # Verify gradients match
        assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
        assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
        assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)


try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
# BATCH, N_HEADS, HEAD_DIM = 4, 32, 64 # To test on A100
BATCH, N_HEADS, HEAD_DIM = 4, 8, 32 # To test on L4
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    if mode == "bwd":
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[2**i for i in range(4, 7)], # will want to check this later
            line_arg="provider",
            line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
            (["flash"] if HAS_FLASH else []),
            line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
            (["Flash-2"] if HAS_FLASH else []),
            styles=[("red", "-"), ("blue", "-"), ("green", "-")],
            ylabel="TFLOPS",
            plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
                "mode": mode,
            },
        ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, mode, provider, device=DEVICE):
    """
    Benchmark different attention implementations
    Compares performance of Triton and Flash Attention implementations
    """
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16

    if "triton" in provider:
        # Benchmark Triton implementation
        q = torch.randn((BATCH, H, N_CTX, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        b = torch.randn((BATCH, H, N_CTX, N_CTX), dtype=dtype, device=device, requires_grad=True)
        # Handle FP8 case
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            # v shoudl be contiguous as B, H, D, N instead of B, H, N, D
            v = v.permute(0, 1, 2, 4, 3).contiguous()
            v = v.permute(0, 1, 2, 4, 3)
            v = v.to(torch.float8_e5m2)
            b = b.to(torch.float8_e5m2)

        sm_scale = 1.3
        fn = lambda: attention(q, k, v, b, sm_scale)

        # Setup backward pass if needed
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)

        ms = triton.testing.do_bench(fn)

    if provider == "flash":
        raise NotImplementedError("Official Flash Attention is not implemented for now")
        # Benchmark Flash Attention implementation
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)

        # Setup backward pass if needed
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)

        ms = triton.testing.do_bench(fn)

    # Calculate FLOPS: TODO: This is still the 1d attention version--need to update for tt
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    # if causal:
    #     total_flops *= 0.5  # Only half the computations needed for causal attention
    if mode == "bwd":
        total_flops *= 2.5  # Additional computations for backward pass

    # Return TFLOPS
    return total_flops * 1e-12 / (ms * 1e-3)

@torch.no_grad()
def reference_tt_attn(q, k, v, b, sm_scale):
    """
    Reference implementation of attention for correctness.
    """
    Z, H, N_CTX = q.shape[:3]
    # Step 1: QK^T. 
    qk = torch.matmul(q, k.transpose(-2, -1))

    # Step 2: Scale and add bias--attn scaling happens before bias addition
    qk_scaled = qk * sm_scale 
    qk_scaled_bias = qk_scaled + b.view(Z, H, 1, N_CTX, N_CTX)

    # Step 3: Softmax
    p = torch.softmax(qk_scaled_bias.float(), dim=-1)

    # Step 4: Output
    ref_out = torch.matmul(p, v)
    return {
        "qk": qk,
        "qk_scaled": qk_scaled,
        "qk_scaled_bias": qk_scaled_bias,
        "p": p,
        "ref_out": ref_out,
    }

@torch.no_grad()
def reference_tt_tiled(q, k, v, b, sm_scale, BLOCK_N=64):
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

    # Initialize tracking tensors for each query position
    m_i = torch.full((Z, H, N_CTX, N_CTX), float('-inf'), device=q.device)  # Max scores
    l_i = torch.ones((Z, H, N_CTX, N_CTX), device=q.device)  # Sum for normalization

    out_dict = {
        "out": out,
        "m_i": m_i,
        "l_i": l_i,
    }

    # Process each batch and head
    for z in range(Z):
        for h in range(H):
            # Process each outer sequence position
            for n1 in range(N_CTX):
                save_debugging = (z == 1) and (h == 2) and (n1 == 4)
                if save_debugging:
                    to_save = {
                        "q_block": [],
                        "k_block": [],
                        "v_block": [],
                        "b_block": [],
                        "qk": [],
                        "qk_scaled": [],
                        "qk_scaled_bias": [],
                        "qk_scaled_bias_minus_max": [],
                        "p": [],
                    }
                # Process blocks of inner sequence
                for start_n in range(0, N_CTX, BLOCK_N):

                    end_n = min(start_n + BLOCK_N, N_CTX)
                    block_size = end_n - start_n

                    # Get current block of query, key, value
                    q_block = q[z, h, n1, :, :]  # [N_CTX, D]
                    k_block = k[z, h, n1, start_n:end_n, :]  # [BLOCK_N, D]
                    v_block = v[z, h, n1, start_n:end_n, :]  # [BLOCK_N, D]
                    b_block = b[z, h, n1, start_n:end_n]  # [BLOCK_N]

                    # Compute attention scores for this block
                    qk = torch.matmul(q_block, k_block.transpose(-2, -1))  # [N_CTX, BLOCK_N]
                    qk_scaled = qk * sm_scale 
                    qk_scaled_bias = qk_scaled + b_block

                    if save_debugging:
                        to_save["q_block"].append(q_block)
                        to_save["k_block"].append(k_block)
                        to_save["v_block"].append(v_block)
                        to_save["b_block"].append(b_block)
                        to_save["qk"].append(qk)
                        to_save["qk_scaled"].append(qk_scaled)
                        to_save["qk_scaled_bias"].append(qk_scaled_bias)

                    # Update running max scores
                    m_ij = torch.max(qk_scaled_bias, dim=-1)[0]  # [N_CTX]
                    m_i_new = torch.maximum(m_i[z, h, n1, :], m_ij)  # [N_CTX]

                    # Compute attention probabilities
                    qk_scaled_bias_minus_max = qk_scaled_bias - m_i_new.unsqueeze(-1)  # Subtract new max for stability
                    p = torch.exp2(qk_scaled_bias_minus_max)  # [N_CTX, BLOCK_N]

                    if save_debugging:
                        to_save["qk_scaled_bias_minus_max"].append(qk_scaled_bias_minus_max)
                        to_save["p"].append(p)
                        out_dict["debugging_intermediate_values"] = to_save

                    # Update sum for normalization
                    l_ij = torch.sum(p, dim=-1)  # [N_CTX]
                    alpha = torch.exp2(m_i[z, h, n1, :] - m_i_new)  # [N_CTX]
                    l_i[z, h, n1, :] = l_i[z, h, n1, :] * alpha + l_ij

                    # Update output accumulator
                    # Scale current output by alpha for stability
                    out[z, h, n1, :, :] = out[z, h, n1, :, :] * alpha.unsqueeze(-1)
                    # Add contribution from current block
                    out[z, h, n1, :, :] = out[z, h, n1, :, :] + torch.matmul(p, v_block)

                    # Update max scores for next iteration
                    m_i[z, h, n1, :] = m_i_new

                # Final normalization
                out[z, h, n1, :, :] = out[z, h, n1, :, :] / l_i[z, h, n1, :].unsqueeze(-1)

    return out_dict

def test_attention_intermediate_values(Z=2, H=4, N_CTX=128, HEAD_DIM=32, BLOCK_N=32, dtype=torch.float32):
    """
    Test function that compares ground truth (full matrix) implementation with CPU-tiled implementation.

    Steps for debugging:
    1. We pick a specific set of indices for B, H, N1:
      - B = 1, H = 2, N1 = 4
      - QN2 ranges (64, 128] -- check second block
      - these (corresponds to a single block)
    2. Run reference implementation, and triton implementation, and check the following:
        for attn_forward_inner, pid = (1, 4, 6) # 1 for N2, 4 for N1, 6 for B
        a. QK matmul for each segment in the loop (BLOCK_N)
            should be exactly the same as reference qk[1, 2, 4, 64:128, start_n:start_n+BLOCK_N]
        b. QK + bias should be exactly the same in the same places
        c. Maximum value of QK should equal whatever was up to that point. For
            the reference implementation, compute current max value of QK on a per-
            block basis.
        d. TODO


    Args:
        Z: Batch size
        H: Number of heads
        N_CTX: Sequence length
        HEAD_DIM: Dimension of each head
        BLOCK_N: Block size for tiled implementation
        dtype: Data type for tensors
    """
    torch.manual_seed(20)

    # Initialize input tensors
    q = torch.randn((Z, H, N_CTX, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
    k = torch.randn((Z, H, N_CTX, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
    v = torch.randn((Z, H, N_CTX, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
    b = torch.ones((Z, H, N_CTX, N_CTX), dtype=dtype, device=DEVICE)
    sm_scale = 0.5

    print("\nComputing CPU-tiled implementation...")
    tiled_out_dict = reference_tt_tiled(q, k, v, b, sm_scale, BLOCK_N)
    tiled_out = tiled_out_dict["out"]
    print(f"Tiled output stats - mean: {tiled_out.mean():.4f}, std: {tiled_out.std():.4f}")

    # Check specific positions
    # Look at a specific batch, head, and sequence position
    b_idx, h_idx, n1_idx = 1, 2, 4  # These match the indices mentioned in the debugging steps
    print(f"\nChecking specific position (b={b_idx}, h={h_idx}, n1={n1_idx}):")
    gt_slice = gt_out[b_idx, h_idx, n1_idx]
    tiled_slice = tiled_out[b_idx, h_idx, n1_idx]
    slice_max_diff = (gt_slice - tiled_slice).abs().max().item()
    slice_mean_diff = (gt_slice - tiled_slice).abs().mean().item()
    print(f"Slice maximum absolute difference: {slice_max_diff:.4f}")
    print(f"Slice mean absolute difference: {slice_mean_diff:.4f}")

    # Compare results
    max_diff = (gt_out - tiled_out).abs().max().item()
    mean_diff = (gt_out - tiled_out).abs().mean().item()
    print(f"\nDifferences between implementations:")
    print(f"Maximum absolute difference: {max_diff:.4f}")
    print(f"Mean absolute difference: {mean_diff:.4f}")

    return {
        "ground_truth": gt_results,
        "tiled_output": tiled_out,
        "max_diff": max_diff,
        "mean_diff": mean_diff
    }




if __name__ == "__main__":
    # Run benchmarks (only works on post-Ampere GPUs)
    # bench_flash_attention.run(save_path=".", print_data=True)
    # test_op(Z=2, H=4, N_CTX=128, HEAD_DIM=32, dtype=torch.float16)
    output_dict = test_attention_intermediate_values()
    breakpoint()