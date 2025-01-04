#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 chunk of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 (sub)matrix (within a block) into 32 warp of shape (32, 4)
 * (which is dealt with by a single thread), then we have
 * a block matrix (of warps) of shape (2 warps, 16 warps).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) (since each thread handles 4) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // TODO: do not modify code, just comment on suboptimal accesses

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    for (; j < end_j; j++)
        // Non-coalesced write--output writes are n apart.
        // Every single one touches its own cache line, so 32 cache lines?
        output[j + n * i] = input[i + n * j];
}


/*
 * Shared memory kernel.
 * Reads global memory in a coalesced way, and then writes it all.
 * We can do this the same way as naive but just with the caching in shared 
 * mem. That is, increment j by 4.
 * Call 
 */
__global__
void shmemTransposeKernel(const float *input, float *output, int n) {

    // TODO: Modify transpose kernel to use shared memory. All global memory
    // reads and writes should be coalesced. Minimize the number of shared
    // memory bank conflicts (0 bank conflicts should be possible using
    // padding). Again, comment on all sub-optimal accesses.

    // __shared__ float data[64 * 64];
    // Bank conflcited version
    // for (; j < end_j; j++) {
    //     data[j + 64 * i] = input[i + n * j];
    // }
    // __syncthreads();
    // for (; j < end_j; j++) {
    //     output[i + n * j] = data[i + 64 * j];
    // }


    // Data needs enough to handle a single block.
    // Adding enough for padding--33/32 for each
    // __shared__ float data[64 * 64];
    __shared__ float data[64 * 64];

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    // // No bank conflict: stride by 33 instead
    // Block reads from data
    for (; j < end_j; j++) {
        data[((j % 64) + 64 * (i % 64))] = input[i + n * j];
        // data[(j + 64 * i) + i] = input[i + n * j];
    }
    __syncthreads();
    // i_out needs to iterate over whatever j was.
    // i_out needs to iterate over whatever i was.
    int i_out = threadIdx.x + 64 * blockIdx.y;
    int j_out = 4 * threadIdx.y + 64 * blockIdx.x;
    int end_j_out = j_out + 4;
    for (; j_out < end_j_out; j_out++) {
        output[j_out * n + i_out] = data[(i_out % 64) + 64 * (j_out % 64)];

        // output[j + n * i] = 999;

    }
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // TODO: This should be based off of your shmemTransposeKernel.
    // Use any optimization tricks discussed so far to improve performance.
    // Consider ILP and loop unrolling.

    // Results:
    // shmem: 0.014336 shmem for 512, 0.574464 for 4096
    // No bank conflict: 0.009216 for 512, 0.577536 for 4096
    // Unrolled: 0.009216 for 512, 0.585728 (??) for 4096
    // ILP

    // Data needs enough to handle a single block.
    // Adding enough for padding--33/32 for each
    // __shared__ float data[64 * 64];
    __shared__ float data[65 * 65];

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    // // No bank conflict: stride by 33 instead
    // Block reads from data
    // With this, we have 0.011264 for size 512, and 0.574464 for size 4096.
    for (; j < end_j; j++) {
        data[((j % 64) + 65 * (i % 64))] = input[i + n * j];
        // data[(j + 64 * i) + i] = input[i + n * j];
    }

    // And now unroll loop
    // data[((j % 64) + 65 * (i % 64))] = input[i + n * j];
    // data[(((j+1) % 64) + 65 * (i % 64))] = input[i + n * (j+1)];
    // data[(((j+2) % 64) + 65 * (i % 64))] = input[i + n * (j+2)];
    // data[(((j+3) % 64) + 65 * (i % 64))] = input[i + n * (j+3)];

    // // And ILP: Haven't finished this yet.
    // int data_start_ind = (j % 64) + 65 * (i % 64);
    // int data_start_ind_2 = data_start_ind + 1;
    // int data_start_ind_3 = data_start_ind + 2;
    // int data_start_ind_4 = data_start_ind + 3;

    // int input_start_ind = i + n * j;
    // int input_start_ind_2 = input_start_ind + n;
    // int input_start_ind_3 = input_start_ind + 2 * n;
    // int input_start_ind_4 = input_start_ind + 3 * n;
    // data[data_start_ind] = input[input_start_ind];
    // data[data_start_ind_2] = input[input_start_ind_2];
    // data[data_start_ind_3] = input[input_start_ind_3];
    // data[data_start_ind_4] = input[input_start_ind_4];
    __syncthreads();
    // i_out needs to iterate over whatever j was.
    // i_out needs to iterate over whatever i was.
    int i_out = threadIdx.x + 64 * blockIdx.y;
    int j_out = 4 * threadIdx.y + 64 * blockIdx.x;
    int end_j_out = j_out + 4;

    for (; j_out < end_j_out; j_out++) {
        output[j_out * n + i_out] = data[(i_out % 64) + 65 * (j_out % 64)];
        // output[j + n * i] = 999;
    }
    // And now unroll loop
    // output[j_out * n + i_out] = data[(i_out % 64) + 65 * (j_out % 64)];
    // output[((j_out+1) * n + i_out)] = data[((i_out % 64) + 65 * ((j_out+1) % 64))];
    // output[((j_out+2) * n + i_out)] = data[((i_out % 64) + 65 * ((j_out+2) % 64))];
    // output[((j_out+3) * n + i_out)] = data[((i_out % 64) + 65 * ((j_out+3) % 64))];

}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}
