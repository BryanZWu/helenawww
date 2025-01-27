// #pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace triangle_attention_gpu_ops
{
    struct TriangleAttentionDescriptor
    {
        int batch_size;
        int seq_len;
        int head_dim;
        int num_heads;
        bool from_starting_node;
    };

    enum TriangleAttentionErrorCodes
    {
        SUCCESS = 0,
        INVALID_ARGUMENT = 1,
        INTERNAL_ERROR = 2,
    };
    struct TriangleAttentionStatus
    {
        TriangleAttentionErrorCodes error_code;
        char *error_msg;
    };

    // Accordign to claude, using the buffers and
    // the opaque to handle inputs and descriptor is
    // a common way to make code framework agnostic.
    void triangle_attention_kernel(
        cudaStream_t stream,
        void **buffers,
        const char *opaque,
        std::size_t opaque_len
    );
    // Args in buffers:
    // 0: q
    // 1: k
    // 2: v
    // 3: mask
    // 4: output

    
    // Backward pass
    void triangle_attention_kernel_backward(
        cudaStream_t stream,
        void **buffers,
        const char *opaque,
        std::size_t opaque_len
    );
} // namespace triangle_attention_gpu_ops
