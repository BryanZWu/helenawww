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
        // then tranpose to (B, num_heads, N, N, head_dim)
        // Initialize cuTENSOR handle
        cutensorHandle_t handle;
        HANDLE_ERROR(cutensorInit(&handle));
        int B = batch_size;
        int N = seq_length;
        int D = num_heads * head_dim;
        int total_size = B * N * N * D;
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

        int32_t q_permutation[5] = {0, 3, 1, 2, 4};
        int32_t k_permutation[5] = {0, 3, 1, 4, 2};

        // Create tensor descriptors
        cutensorTensorDescriptor_t desc_in, desc_out;
        HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &desc_in,
                                                  4, dims_in, strides_in, CUDA_R_32F, 0));
        HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &desc_out,
                                                  5, dims_out, strides_out, CUDA_R_32F, 0));

        // Allocate temporary memory for the transposed result
        float *q_tranposed;
        float *k_tranposed;
        size_t qk_mem_size = total_size * sizeof(float);
        cudaMalloc(&q_output, qk_mem_size);
        cudaMalloc(&k_tranposed, qk_mem_size);
        // Perform the q and k transpose
        // cutensorPermutation handles the dimensionality difference internally,
        // as long as the total number of elements remains the same
        HANDLE_ERROR(cutensorPermutation(&handle,
                                         q, &desc_in, q_permutation,
                                         q_tranposed, &desc_out,
                                         CUDA_R_32F, 0)); // No stream specified
        HANDLE_ERROR(cutensorPermutation(&handle,
                                        k, &desc_in, k_permutation,
                                        k_tranposed, &desc_out,
                                        CUDA_R_32F, 0)); // No stream specified




        //matmul attn = q @ k
        // Initialize cuTENSOR handle

            // Input and output tensor dimensions
            int64_t dims_qt[5] = {B_dim, num_heads, N_dim, N_dim, head_dim};  // (B, num_heads, N, N, head_dim)
            int64_t dims_kt[5] = {B_dim, num_heads, N_dim, head_dim, N_dim};  // (B, num_heads, N, head_dim, N)
            int64_t dims_attn[5] = {B_dim, num_heads, N_dim, N_dim, N_dim};     // (B, num_heads, N, N, N)

            // Strides for row-major layout
            int64_t strides_qt[5];
            strides_A[4] = 1;
            for (int i = 3; i >= 0; --i)
                strides_A[i] = strides_A[i + 1] * dims_A[i + 1];

            int64_t strides_kt[5];
            strides_B[4] = 1;
            for (int i = 3; i >= 0; --i)
                strides_B[i] = strides_B[i + 1] * dims_B[i + 1];

            int64_t strides_attn[5];
            strides_C[4] = 1;
            for (int i = 3; i >= 0; --i)
                strides_C[i] = strides_C[i + 1] * dims_C[i + 1];

            // Modes for each tensor (label dimensions)
            int32_t modes_qt[5] = {'b', 'h', 'i', 'j', 'd'};  // A: (B, num_heads, N, N, head_dim)
            int32_t modes_kt[5] = {'b', 'h', 'i', 'd', 'k'};  // B: (B, num_heads, N, head_dim, N)
            int32_t modes_attn[5] = {'b', 'h', 'i', 'j', 'k'};  // C: (B, num_heads, N, N, N)

            // Tensor descriptors
            cutensorTensorDescriptor_t desc_A, desc_B, desc_C;
            HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &desc_A,
                                                    5, dims_A, strides_A, CUDA_R_32F, 0));
            HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &desc_B,
                                                    5, dims_B, strides_B, CUDA_R_32F, 0));
            HANDLE_ERROR(cutensorInitTensorDescriptor(&handle, &desc_C,
                                                    5, dims_C, strides_C, CUDA_R_32F, 0));

            // Contraction descriptor
            cutensorContractionDescriptor_t desc;
            HANDLE_ERROR(cutensorInitContractionDescriptor(&handle, &desc,
                                                        &desc_A, modes_A,
                                                        &desc_B, modes_B,
                                                        &desc_C, modes_C,
                                                        &desc_C, modes_C,
                                                        CUDA_R_32F));

            // Find workspace size
            size_t workspace_size;
            HANDLE_ERROR(cutensorContractionGetWorkspaceSize(&handle, &desc, CUTENSOR_ALGO_DEFAULT, &workspace_size));

            // Allocate workspace
            void* workspace = nullptr;
            if (workspace_size > 0) {
                cudaMalloc(&workspace, workspace_size);
            }

            // Alpha and beta scalars
            float alpha = 1.0f;
            float beta = 0.0f;

            // Perform tensor contraction
            HANDLE_ERROR(cutensorContraction(&handle, &desc,
                                            &alpha, A, B,
                                            &beta, C, C,
                                            workspace, workspace_size, 0));  // No CUDA stream

            // Free workspace
            if (workspace) {
                cudaFree(workspace);
            }
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
