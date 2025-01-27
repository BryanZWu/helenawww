#ifndef TRIANGLE_ATTENTION_H_
#define TRIANGLE_ATTENTION_H_

#include <cuda_runtime.h>

// Descriptor struct to hold kernel parameters
struct TriangleAttentionDescriptor
{
    int batch_size;
    int seq_length;
    int num_heads;
    int head_dim;
    bool from_starting_node;
};

// C interface for kernel launch
extern "C"
{
    // Launch the triangle attention CUDA kernel
    cudaError_t LaunchTriangleAttention(
        float *q,      // [batch_size, seq_len, seq_len, hidden_dim]
        float *k,      // [batch_size, seq_len, seq_len, hidden_dim]
        float *v,      // [batch_size, seq_len, seq_len, hidden_dim]
        float *mask,   // [batch_size * num_heads, seq_len, seq_len, seq_len]
        float *output, // [batch_size, seq_len, seq_len, hidden_dim]
        const TriangleAttentionDescriptor *desc,
        cudaStream_t stream);
}

#endif // TRIANGLE_ATTENTION_H_
