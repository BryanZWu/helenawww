#include "triangle_attention.h"
#include <cuda_runtime.h>
// #include <cutensor.h>
#define HANDLE_ERROR(x)                                                                \
    {                                                                                  \
        if ((x) != CUTENSOR_STATUS_SUCCESS)                                            \
        {                                                                              \
            std::cerr << "cuTENSOR error: " << cutensorGetErrorString(x) << std::endl; \
            exit(1);                                                                   \
        }                                                                              \
    }
namespace
{

    // CUDA kernel declaration
    __global__ void triangle_attention_kernel(
        const float *q,
        const float *k,
        const float *v,
        const float *mask,
        float *output,
        const int batch_size,
        const int seq_length,
        const int num_heads,
        const int head_dim,
        const bool from_starting_node)
    {
        // TODO: Implement the actual kernel
        // reshape q and k shape (B, N, N, D) to (B, N, N, num_heads, head_dim)
        // tranpose to (B, num_heads, N, N, head_dim)
        // Initialize cuTENSOR handle
        cutensorHandle_t handle;
        HANDLE_ERROR(cutensorInit(&handle));
        int B = batch_size;
        int N = seq_length;
        int D = num_heads * head_dim;
        // Input tensor dimensions and strides
        int64_t dims_in[4] = {B, N, N, D};
        int64_t strides_in[4];
        strides_in[3] = 1;
        strides_in[2] = D;
        strides_in[1] = N * D;
        strides_in[0] = N * N * D;

        // Output tensor dimensions and strides
        int64_t dims_out[5] = {B, num_heads, N, N, head_dim};
        int64_t strides_out[5];
        strides_out[4] = 1;
        strides_out[3] = head_dim;
        strides_out[2] = num_heads * head_dim;
        strides_out[1] = N * num_heads * head_dim;
        strides_out[0] = N * N * num_heads * head_dim;

        // Define the permutation: (0, 3, 1, 2, 4)
        int32_t permutation[5] = {0, 3, 1, 2, 4};

        // Create tensor descriptors
        cutensorTensorDescriptor_t desc_in, desc_out;
        HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &desc_in,
                                                  4, dims_in, strides_in, CUDA_R_32F, 0));
        HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &desc_out,
                                                  5, dims_out, strides_out, CUDA_R_32F, 0));

        // Perform the transpose
        HANDLE_ERROR(cutensorPermutation(&handle,
                                         q, &desc_in, permutation,
                                         output, &desc_out,
                                         CUDA_R_32F, 0)); // No stream specified
    }
    // TODO: potential backward pass

} // namespace

extern "C"
{

    cudaError_t LaunchTriangleAttention(
        float *q,
        float *k,
        float *v,
        float *mask,
        float *output,
        const TriangleAttentionDescriptor *desc,
        cudaStream_t stream)
    {
        // Calculate grid and block dimensions
        dim3 block(256); // TODO: Optimize these dimensions
        dim3 grid((desc->seq_length * desc->seq_length + block.x - 1) / block.x);

        // Launch kernel
        triangle_attention_kernel<<<grid, block, 0, stream>>>(
            q, k, v, mask, output,
            desc->batch_size,
            desc->seq_length,
            desc->num_heads,
            desc->head_dim,
            desc->from_starting_node);

        return cudaGetLastError();
    }
}
