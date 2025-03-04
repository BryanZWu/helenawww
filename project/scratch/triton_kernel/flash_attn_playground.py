import pytest
import torch

import triton
import triton.language as tl

# Environment variable explanation for Blackwell architecture
# ENABLE_LHS_TO_TMEM is an experimental environment variable for Blackwell.
# If set to 1, it can improve performance of Blackwell attention, but may cause
# correctness issues outside of the _attn_fwd_tma kernel

# Get the active torch device (GPU)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Helper function to check if we're using HIP (AMD) backend
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

# Helper function to check if we're using CUDA backend
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

# Helper function to check if TMA (Tensor Memory Access) is supported
# Requires CUDA and GPU capability >= 9
def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

# Check if TMA descriptor type is available in Triton language
HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)

# Print information about TMA benchmark configuration
if HAS_TMA_DESC:
    print("TMA benchmarks will be running with experimental grid constant TMA descriptor.")
else:
    print("TMA benchmarks will be running without grid constant TMA descriptor.")

# Helper class for TMA (Tensor Memory Access) auto-tuning
class TmaAutoTuneHelper:
    # Inner class to wrap kernel parameters for TMA descriptors
    class KernelParamWrapper:
        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    # Constants
    TMA_SIZE = 128  # Size of TMA descriptor

    def __init__(self):
        # Initialize TMA descriptor utility functions
        self.fill_1d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_1d_tma_descriptor)
        self.fill_2d_tma_descriptor_inner = (triton.runtime.driver.active.utils.fill_2d_tma_descriptor)
        # Initialize descriptor storage based on TMA availability
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Initialize TMA descriptor with given name
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8)
        else:
            self.cuda_descriptors[name] = torch.empty(TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8)

    # Fill 1D TMA descriptor with given parameters
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0  # Ensure 64-byte alignment
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    # Fill 2D TMA descriptor with given parameters
    def fill_2d_tma_descriptor(self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0  # Ensure 64-byte alignment
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr())
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr())
            desc_x.copy_(buf_x, non_blocking=True)

    # Get TMA descriptor kernel parameter
    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    """
    Inner kernel for attention forward pass computation
    Computes attention scores and updates accumulator for a block of the attention matrix
    """
    # Determine range of values for current processing stage
    if STAGE == 1:
        # Pre-mask stage: process elements before the mask
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # Mask stage: process elements within the mask
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)  # Ensure alignment
    else:
        # Post-mask stage: process all remaining elements
        lo, hi = 0, N_CTX
        
    # Advance block pointers to correct starting positions
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    
    # Process blocks of K and V to compute attention
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Compute attention scores (Q * K^T)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        
        # Apply causal mask if in mask stage
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            # Regular softmax scaling
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            
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
        
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                        desc_k, desc_v,  #
                        offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                        STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                        N_CTX: tl.constexpr):
    """
    TMA (Tensor Memory Access) version of the attention forward pass kernel
    Uses TMA descriptors for more efficient memory access patterns
    Similar to _attn_fwd_inner but optimized for TMA operations
    """
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetkv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl._experimental_descriptor_load(desc_k, [offsetkv_y, 0], [BLOCK_N, HEAD_DIM], dtype).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl._experimental_descriptor_load(desc_v, [offsetkv_y, 0], [BLOCK_N, HEAD_DIM], dtype)
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        offsetkv_y += BLOCK_N
    return acc, l_i, m_i


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


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  # sm_scale: scaling factor for softmax, M: max values, Out: output tensor
              stride_qz, stride_qh, stride_qm, stride_qk,  # Strides for Q tensor: batch, head, seq_len, head_dim
              stride_kz, stride_kh, stride_kn, stride_kk,  # Strides for K tensor
              stride_vz, stride_vh, stride_vk, stride_vn,  # Strides for V tensor
              stride_oz, stride_oh, stride_om, stride_on,  # Strides for Output tensor
              Z, H, N_CTX,  # Z=batch_size, H=num_heads, N_CTX=sequence_length
              HEAD_DIM: tl.constexpr,  # Size of each attention head
              BLOCK_M: tl.constexpr,  # Block size for sequence dimension
              BLOCK_N: tl.constexpr,  # Block size for attention computation
              STAGE: tl.constexpr  # Determines attention pattern (1=non-causal, 3=causal)
              ):
    # Ensure block size doesn't exceed head dimension
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    
    # Grid dimension explanation:
    # - program_id(0): Indexes along sequence length (N_CTX/BLOCK_M blocks)
    # - program_id(1): Indexes along batch*heads dimension (Z*H blocks)
    start_m = tl.program_id(0)  # Block index in sequence dimension
    off_hz = tl.program_id(1)   # Combined batch and head index
    
    # Convert combined batch-head index into separate indices
    off_z = off_hz // H         # Batch index
    off_h = off_hz % H          # Head index
    
    # Calculate base offset for current batch and head
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # Create block pointers for efficient memory access:
    # Q: Current block of queries [BLOCK_M, HEAD_DIM]
    # K: Current block of keys [HEAD_DIM, BLOCK_N]
    # V: Current block of values [BLOCK_N, HEAD_DIM]
    # O: Output block [BLOCK_M, HEAD_DIM]
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # Initialize per-block data structures:
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Sequence offsets within block
    offs_n = tl.arange(0, BLOCK_N)                      # Attention offsets within block
    
    # Initialize tracking variables for softmax computation
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Max values for stability
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0          # Sum for normalization
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)      # Accumulated attention outputs

    # Scale factor for attention scores (includes log2 conversion for numerical stability)
    qk_scale = sm_scale * 1.44269504  # 1.44269504 = 1/log(2)

    # Load query block - remains in SRAM throughout computation
    q = tl.load(Q_block_ptr)

    # Process attention in two stages:
    # Stage 1 (STAGE & 1): Process "off-band" attention
    # - For causal attention: handles all positions before the current block
    # - For non-causal attention: handles all positions
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # Stage 2 (STAGE & 2): Process "on-band" attention
    # - For causal attention: handles current block with masking
    # - For non-causal attention: skipped
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # Final processing:
    # 1. Apply softmax normalization
    # 2. Store softmax scaling factors (m_i) for backward pass
    # 3. Store final output values
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
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


def keep_tma(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if (torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8):
        return False
    return True


@triton.autotune(configs=list(filter(keep_tma, configs_tma)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"])
@triton.jit
def _attn_fwd_tma(sm_scale, M,  #
                  Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                  HEAD_DIM: tl.constexpr,  #
                  BLOCK_M: tl.constexpr,  #
                  BLOCK_N: tl.constexpr,  #
                  FP8_OUTPUT: tl.constexpr,  #
                  STAGE: tl.constexpr  #
                  ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offset_y = off_z + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl._experimental_descriptor_load(desc_q, [qo_offset_y, 0], [BLOCK_M, HEAD_DIM], dtype)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                                            desc_k, desc_v,  #
                                            offset_y, dtype, start_m, qk_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            4 - STAGE, offs_m, offs_n, N_CTX,  #
                                            )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                                            desc_k, desc_v,  #
                                            offset_y, dtype, start_m, qk_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            2, offs_m, offs_n, N_CTX,  #
                                            )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl._experimental_descriptor_store(desc_o, acc.to(dtype), [qo_offset_y, 0])


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    """
    Preprocesses data for attention backward pass
    Computes initial delta values needed for gradient computation
    """
    # Calculate offsets for the current block
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    
    # Load output and gradient tensors
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    
    # Compute delta values (dot product of output and gradient)
    delta = tl.sum(o * do, axis=1)
    
    # Store computed deltas
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, sm_scale,  #
                   DO,  #
                   M, D,  #
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   start_n, start_m, num_steps,  #
                   MASK: tl.constexpr):
    """
    Computes gradients with respect to keys (dk) and values (dv)
    Main computation kernel for the backward pass
    """
    # Initialize offsets for the current block
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # Setup pointers for Q and DO tensors
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    
    # Ensure block sizes are compatible
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    
    # Initialize tracking variables
    curr_m = start_m
    step_m = BLOCK_M1
    
    # Process blocks to compute gradients
    for blk_idx in range(num_steps):
        # Load Q values and compute Q*K^T
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        
        # Compute attention probabilities
        pT = tl.math.exp2(qkT - m[None, :])
        
        # Apply causal masking if needed
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
            
        # Load gradients
        do = tl.load(do_ptrs)
        
        # Compute dV gradients
        ppT = pT.to(tl.float16)
        dv += tl.dot(ppT, do)
        
        # Load precomputed deltas
        Di = tl.load(D + offs_m)
        
        # Compute dP (gradient of probabilities) and dS (gradient of scores)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        
        # Compute dK gradients
        dk += tl.dot(dsT, tl.trans(qT))
        
        # Update pointers for next iteration
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
        
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,  #
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,  #
                 start_m, start_n, num_steps,  #
                 MASK: tl.constexpr):
    """
    Computes gradients with respect to queries (dq)
    Part of the backward pass that handles query gradient computation
    """
    # Initialize offsets for the current block
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    
    # Setup pointers for K and V tensors
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    
    # Load precomputed deltas
    Di = tl.load(D + offs_m)
    
    # Ensure block sizes are compatible
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    
    # Initialize tracking variables
    curr_n = start_n
    step_n = BLOCK_N2
    
    # Process blocks to compute gradients
    for blk_idx in range(num_steps):
        # Load K and V values
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        
        # Compute attention scores
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        
        # Apply causal masking if needed
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
            
        # Compute gradients
        dp = tl.dot(do, vT).to(tl.float32)  # Gradient wrt attention probabilities
        ds = p * (dp - Di[:, None])  # Gradient wrt attention scores
        ds = ds.to(tl.float16)
        
        # Update query gradients
        dq += tl.dot(ds, tl.trans(kT))
        
        # Update pointers for next iteration
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
        
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(dk, dv,  #
                            Q, k, v, sm_scale,  #
                            DO,  #
                            M, D,  #
                            stride_tok, stride_d,  #
                            H, N_CTX,  #
                            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
                            start_n, start_m, num_steps,  #
                            MASK=True  #
                            )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, k, v, sm_scale,  #
        DO,  #
        M, D,  #
        stride_tok, stride_d,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps,  #
        MASK=False  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
                      MASK=True  #
                      )
    end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n - num_steps * BLOCK_N2, num_steps,  #
                      MASK=False  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class _attention(torch.autograd.Function):
    """
    Custom PyTorch autograd function for flash attention implementation
    Implements efficient forward and backward passes for the attention mechanism
    """
    
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, USE_TMA=True):
        """
        Forward pass of flash attention
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            causal: Whether to use causal attention
            sm_scale: Softmax scaling factor
            USE_TMA: Whether to use Tensor Memory Access optimization
        """
        # Verify shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]  # V may be transposed for float8
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        
        # Initialize output tensor
        o = torch.empty_like(q)
        
        # Determine attention stage based on causality
        stage = 3 if causal else 1
        
        # Setup kernel arguments
        extra_kern_args = {}
        
        # Special tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        # Initialize scaling factor storage
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        
        # Choose between TMA and regular implementation
        if USE_TMA and supports_tma() and not (torch.cuda.get_device_capability()[0] == 9
                                               and q.dtype == torch.float8_e5m2):
            # TMA implementation path
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]
            
            # Initialize TMA descriptors
            desc_helper = TmaAutoTuneHelper()
            desc_helper.init_tma_descriptor("q")
            desc_helper.init_tma_descriptor("v")
            desc_helper.init_tma_descriptor("k")
            desc_helper.init_tma_descriptor("o")
            
            # Grid function for TMA kernel
            def grid(META):
                nonlocal desc_helper
                
                # Fill TMA descriptors for all tensors
                desc_helper.fill_2d_tma_descriptor("q", q.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_M"], HEAD_DIM_K,
                                                   q.element_size())
                desc_helper.fill_2d_tma_descriptor("v", v.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_N"], HEAD_DIM_K,
                                                   v.element_size())
                desc_helper.fill_2d_tma_descriptor("k", k.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_N"], HEAD_DIM_K,
                                                   k.element_size())
                desc_helper.fill_2d_tma_descriptor("o", o.data_ptr(), y_dim, HEAD_DIM_K, META["BLOCK_M"], HEAD_DIM_K,
                                                   o.element_size())
                
                return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

            # Get TMA descriptors
            desc_q = desc_helper.get_tma_descriptor_kernel_param("q")
            desc_v = desc_helper.get_tma_descriptor_kernel_param("v")
            desc_k = desc_helper.get_tma_descriptor_kernel_param("k")
            desc_o = desc_helper.get_tma_descriptor_kernel_param("o")
            
            # Store grid function for backward pass
            ctx.grid = grid
            
            # Launch TMA kernel
            _attn_fwd_tma[grid](
                sm_scale, M,
                q.shape[0], q.shape[1],
                desc_q, desc_k, desc_v, desc_o,
                N_CTX=q.shape[2],
                HEAD_DIM=HEAD_DIM_K,
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,
                STAGE=stage,
                **extra_kern_args)
        else:
            # Regular implementation path
            def grid(args):
                return (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
            
            ctx.grid = grid
            
            # Launch regular kernel
            _attn_fwd[grid](
                q, k, v, sm_scale, M, o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                q.shape[0], q.shape[1],
                N_CTX=q.shape[2],
                HEAD_DIM=HEAD_DIM_K,
                STAGE=stage,
                **extra_kern_args)

        # Save tensors needed for backward pass
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        """
        Backward pass of flash attention
        Computes gradients with respect to query, key and value tensors
        Args:
            do: Gradient of the output tensor
        """
        # Retrieve saved tensors from forward pass
        q, k, v, o, M = ctx.saved_tensors
        
        # Verify gradient tensor properties
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        
        # Initialize gradient tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Get tensor dimensions
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        
        # Define block sizes and parameters
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 5
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        
        # Scale key tensor for gradient computation
        arg_k = k * (ctx.sm_scale * RCP_LN2)
        
        # Verify context size is compatible with block size
        assert N_CTX % PRE_BLOCK == 0
        
        # Define preprocessing grid dimensions
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        
        # Initialize delta tensor for gradient computation
        delta = torch.empty_like(M)
        breakpoint()
        
        # Preprocess gradients
        _attn_bwd_preprocess[pre_grid](
            o, do,
            delta,
            BATCH, N_HEAD, N_CTX,
            BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM
        )
        
        # Define main backward pass grid dimensions
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        
        # Compute gradients
        _attn_bwd[grid](
            q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,
            M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            N_HEAD, N_CTX,
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=ctx.HEAD_DIM,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES
        )

        return dq, dk, dv, None, None


attention = _attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    """
    Test function for attention implementation
    Compares Triton implementation with PyTorch reference implementation
    """
    # Set random seed for reproducibility
    torch.manual_seed(20)
    
    # Initialize input tensors
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    dout = torch.randn_like(q)
    
    # Reference implementation using PyTorch
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))  # Causal mask
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale  # Compute attention scores
    if causal:
        p[:, :, M == 0] = float("-inf")  # Apply causal mask
    p = torch.softmax(p.float(), dim=-1).half()  # Apply softmax
    ref_out = torch.matmul(p, v)  # Compute attention output
    
    # Compute reference gradients
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    
    # Triton implementation
    tri_out = attention(q, k, v, causal, sm_scale).half()
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    
    # Compare results
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
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
BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [True, False]:
        if mode == "bwd" and not causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(10, 15)],
                line_arg="provider",
                line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                (["flash"] if HAS_FLASH else []),
                line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                (["Flash-2"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="TFLOPS",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device=DEVICE):
    """
    Benchmark different attention implementations
    Compares performance of Triton and Flash Attention implementations
    """
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16
    
    if "triton" in provider:
        # Benchmark Triton implementation
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        
        # Handle FP8 case
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
            
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale)
        
        # Setup backward pass if needed
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
            
        ms = triton.testing.do_bench(fn)
        
    if provider == "flash":
        # Benchmark Flash Attention implementation
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        
        # Setup backward pass if needed
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
            
        ms = triton.testing.do_bench(fn)
    
    # Calculate FLOPS
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5  # Only half the computations needed for causal attention
    if mode == "bwd":
        total_flops *= 2.5  # Additional computations for backward pass
        
    # Return TFLOPS
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    # Initialize CUDA context before running tests
    if torch.cuda.is_available():
        torch.cuda.init()
        # # Optional: you can also explicitly set the device
        # torch.cuda.set_device(0)
    
    # Run test
    print("Running attention test...")
    test_op(Z=1, H=2, N_CTX=1024, HEAD_DIM=64, causal=True)
    print("Test completed successfully!")

    # # Run benchmarks (only works on post-Ampere GPUs)
    # print("\nRunning benchmarks...")
    # bench_flash_attention.run(save_path=".", print_data=True)

