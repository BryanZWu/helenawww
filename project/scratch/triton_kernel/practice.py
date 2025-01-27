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

def matmul_kernel(
    A,
    B,
    C, m, n, k, block_size: tl.constexpr,
):
    pass

if __name__ == "__main__":
    test_add_kernel()

