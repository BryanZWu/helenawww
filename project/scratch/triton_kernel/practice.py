# Practice triton kernel
import torch

import triton
import triton.language as tl


DEVICE = torch.device("cuda")

@triton.jit
def add_kernel(
    X1,
    X2,
    Y,
    num_elements,
    block_size: tl.constexpr,
):
    # for an add kernel, in CUDA we need to
    # determine block # block size and add
    # one thread at a time. And then we increment
    # to the next block using each thread via
    # a loop. 

    
    # But for here, we simply do the pointer 
    # arithmetic? 
    indices = tl.arange(0, block_size)
    block_id = tl.program_id(axis=0)
    total_offset = block_id * block_size + indices
    mask  = total_offset < num_elements
    vals1 = tl.load(X1 + total_offset, mask=mask)
    vals2 = tl.load(X2 + total_offset, mask=mask)
    tl.store(Y + total_offset, vals1 + vals2, mask=mask)

def test_add_kernel():
    X1 = torch.randn(1000, device=DEVICE)
    X2 = torch.randn(1000, device=DEVICE)
    Y = torch.empty_like(X1)
    # Grid is the grid size, as in cuda. We specify
    # the grid size based on what's needed from "meta"
    grid = lambda meta: (triton.cdiv(Y.numel(), meta['block_size']), print(meta.keys()))

    add_kernel[grid](X1, X2, Y, 1000, 128)
    Y_expected = X1 + X2
    assert torch.allclose(Y, Y_expected)

def softmax_kernel(
    X,
    Y,
    n_rows,
    n_cols,
    block_size: tl.constexpr,
):
    # Version 1: assume that row and column is continuous--will 
    # modify later to remove that assumption
    block_id = tl.program_id(axis=0)
    # rows_per_block = block_size / cols # or something
    num_blocks = tl.num_programs(axis=0)
    for offset in tl.arange(block_id, n_rows, num_blocks, num_stages=4):
        data = tl.load(X + offset, n_cols)
        data_max = tl.max(data)
        data = data - data_max # vectorized, love that
        exp_data = tl.exp(data)
        output_data = exp_data / tl.sum(exp_data)
        tl.store(Y + offset, output_data)

# @triton.jit
def matmul_kernel(
    A, B, C,
    m, n, k,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    block_size_m: tl.constexpr, block_size_n: tl.constexpr, 
    block_size_k: tl.constexpr,
):
    """
    Drafting out what this would look like: 
    First pass will involve no L2 cache optimization for now. 
    First pass: 
    - For each block_size m by n, use a single pid for it.
    - that block size iterates throguh all k that's necessary. This 
        is the fastest thing to do sicne it's a single write after 
        shared memory accumulation.
    - Use the 2d stuff within a single block to handle the local matmul
    
    second pass:
    - Proper masking
    - Reordering for L2 cache optim
    """
    block_id = tl.program_id(0)
    num_blocks_m = tl.cdiv(m, block_size_m)
    num_blocks_n = tl.cdiv(n, block_size_n)
    num_blocks_k = tl.cdiv(k, block_size_k)
    block_id_m = block_id // num_blocks_n
    block_id_n = block_id % num_blocks_n
    block_m_start = block_id_m * block_size_m
    block_n_start = block_id_n * block_size_n
    idx_am = (tl.arange(0, block_size_m) + block_m_start) * stride_am
    idx_bn = (tl.arange(0, block_size_n) + block_n_start) * stride_bn
    idx_cm = (tl.arange(0, block_size_m) + block_m_start) * stride_cm
    idx_cn = (tl.arange(0, block_size_n) + block_n_start) * stride_cn

    # TODO: need to handle the masking--for now let's not?

    accumulator = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)

    # Now loop over the ks
    for i in range(num_blocks_k):
        idx_ak = (tl.arange(0, block_size_k) + i * block_size_k) * stride_ak
        idx_bk = (tl.arange(0, block_size_k) + i * block_size_k) * stride_bk
        idx_a = idx_am[:, None] + idx_ak[None, :] + A
        idx_b = idx_bk[:, None] + idx_bn[None, :] + B
        data_a = tl.load(idx_a)
        data_b = tl.load(idx_b)
        # accumulator = tl.dot(data_a, data_b, accumulator)
        accumulator += tl.dot(data_a, data_b)
    idx_c = idx_cm[:, None] + idx_cn[None, :] + C
    tl.store(idx_c, accumulator)

@triton.jit
def matmul_kernel_v2(
    A, B, C,
    m, n, k,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    block_size_m: tl.constexpr, block_size_n: tl.constexpr, 
    block_size_k: tl.constexpr,
):
    """
    Drafting out what this would look like: 
    First pass will involve no L2 cache optimization for now. 
    First pass: 
    - For each block_size m by n, use a single pid for it.
    - that block size iterates throguh all k that's necessary. This 
        is the fastest thing to do sicne it's a single write after 
        shared memory accumulation.
    - Use the 2d stuff within a single block to handle the local matmul
    
    second pass:
    - Proper masking
    - Reordering for L2 cache optim
    """
    block_id = tl.program_id(0)
    num_blocks_m = tl.cdiv(m, block_size_m)
    num_blocks_n = tl.cdiv(n, block_size_n)
    num_blocks_k = tl.cdiv(k, block_size_k)
    block_id_m = block_id // num_blocks_n
    block_id_n = block_id % num_blocks_n
    block_m_start = block_id_m * block_size_m
    block_n_start = block_id_n * block_size_n

    idx_am_prestride = (tl.arange(0, block_size_m) + block_m_start)
    mask_am = idx_am_prestride < m
    idx_am = idx_am_prestride * stride_am

    idx_bn_prestride = (tl.arange(0, block_size_n) + block_n_start)
    mask_bn = idx_bn_prestride < n
    idx_bn = idx_bn_prestride * stride_bn

    idx_cm_prestride = (tl.arange(0, block_size_m) + block_m_start)
    mask_cm = idx_cm_prestride < m
    idx_cm = idx_cm_prestride * stride_cm

    idx_cn_prestride = (tl.arange(0, block_size_n) + block_n_start)
    mask_cn = idx_cn_prestride < n
    idx_cn = idx_cn_prestride * stride_cn

    # TODO: need to handle the masking--for now let's not?

    accumulator = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)

    # Now loop over the ks
    for i in range(num_blocks_k):
        idx_k_prestride = (tl.arange(0, block_size_k) + i * block_size_k)
        mask_k = idx_k_prestride < k
        idx_ak = idx_k_prestride * stride_ak
        idx_bk = idx_k_prestride * stride_bk
        idx_a = A + idx_am[:, None] + idx_ak[None, :]
        idx_b = B + idx_bk[:, None] + idx_bn[None, :]
        mask_a = mask_am[:, None] & mask_k[None, :]
        mask_b = mask_k[:, None] & mask_bn[None, :]
        data_a = tl.load(idx_a, mask=mask_a, other=0)
        data_b = tl.load(idx_b, mask=mask_b, other=0)
        accumulator += tl.dot(data_a, data_b)
        # accumulator = tl.dot(data_a, data_b, accumulator)
    idx_c = C + idx_cm[:, None] + idx_cn[None, :]
    mask_c = mask_cm[:, None] & mask_cn[None, :]
    tl.store(idx_c, accumulator, mask=mask_c)

def test_matmul_kernel():
    # M, N, K = 1024, 1024, 1024  # Using power of 2 dimensions for simplicity
    M = N = K = 1024
    A = torch.randn(M, K, device=DEVICE)
    B = torch.randn(K, N, device=DEVICE)
    C = torch.empty(M, N, device=DEVICE)

    # Configure block sizes - these are typically power of 2 values
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    # Grid is the number of blocks needed to cover the output matrix
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    # Launch kernel with proper strides and dimensions
    matmul_kernel_v2[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),  # stride_am, stride_ak
        B.stride(0), B.stride(1),  # stride_bk, stride_bn
        C.stride(0), C.stride(1),  # stride_cm, stride_cn
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )

    # Verify result
    C_expected = A @ B
    max_diff = (C - C_expected).abs().max().item()
    print(f"Maximum absolute difference: {max_diff}")
    
    # Use a more appropriate tolerance for GPU floating point operations
    assert torch.allclose(C, C_expected, rtol=1e-3, atol=1e-3), f"Max difference of {max_diff} exceeds tolerance"
    print("Matrix multiplication test passed!")

# Benchmark configurations
configs = [
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as x-axis
        x_vals=[512, 1024, 2048, 4096],  # Different values for M, N, K
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # Possible values for that argument
        line_names=['Triton', 'Torch'],  # Label name for the lines
        styles=[('blue', '-'), ('red', '-')],  # Line styles
        ylabel='TFLOPS',  # Label name for the y-axis
        plot_name='matmul-performance',  # Name for the plot
        args={},  # Values for other arguments not in x_names and line_arg
    )
]

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    # Create random tensors
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float32)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float32)
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        def run_triton():
            c = torch.empty((M, N), device=DEVICE, dtype=torch.float32)
            matmul_kernel_v2[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            )
            return c
        ms, min_ms, max_ms = triton.testing.do_bench(run_triton, quantiles=quantiles)
    
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    # test_matmul_kernel()
    # test_add_kernel()
    benchmark.run(show_plots=True, print_data=True)

