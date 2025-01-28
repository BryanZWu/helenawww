#include "triangle_attention.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

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
        cg::thread_block block = cg::this_thread_block();

        // Indices for batch, head, and sequence
        int batch_idx = blockIdx.x;
        int head_idx = blockIdx.y;
        int seq_idx = threadIdx.x;

        // Shared memory to store intermediate results
        __shared__ float shared_q[];  // dont use 64, use dynamic allocation when calling kerneles
        __shared__ float shared_k[];
        __shared__ float shared_v[];
        __shared__ float shared_logits[]; // For attention logits
        __shared__ float shared_scores[]; // For attention scores

        // Global memory offsets
        int q_offset = ((batch_idx * num_heads + head_idx) * seq_length + seq_idx) * head_dim;
        int k_offset = ((batch_idx * num_heads + head_idx) * seq_length + seq_idx) * head_dim;
        int v_offset = ((batch_idx * num_heads + head_idx) * seq_length + seq_idx) * head_dim;
        int mask_offset = batch_idx * seq_length * seq_length + seq_idx * seq_length;

        // Load Q, K, V into shared memory
        if (seq_idx < head_dim) {
            shared_q[seq_idx] = q[q_offset + seq_idx];
            shared_k[seq_idx] = k[k_offset + seq_idx];
            shared_v[seq_idx] = v[v_offset + seq_idx];
        }
        block.sync();

        // Compute attention logits
        float attn_logit = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            attn_logit += shared_q[i] * shared_k[i];
        }
        attn_logit /= sqrtf((float)head_dim);

        // Apply mask (if available)
        if (mask != nullptr) {
            attn_logit += mask[mask_offset + seq_idx] == 0 ? -1e9 : 0;
        }
        shared_logits[seq_idx] = attn_logit;
        block.sync();

        // Softmax computation
        float max_logit = -1e9;
        for (int i = 0; i < seq_length; i++) {
            max_logit = fmaxf(max_logit, shared_logits[i]);
        }
        block.sync();

        float sum_exp = 0.0f;
        for (int i = 0; i < seq_length; i++) {
            shared_scores[i] = expf(shared_logits[i] - max_logit);
            sum_exp += shared_scores[i];
        }
        block.sync();

        for (int i = 0; i < seq_length; i++) {
            shared_scores[i] /= sum_exp;
        }
        block.sync();

        // Compute attention output
        float attn_out = 0.0f;
        for (int i = 0; i < seq_length; i++) {
            attn_out += shared_scores[i] * shared_v[i];
        }
        block.sync();

        // Write output to global memory
        if (seq_idx < head_dim) {
            output[q_offset + seq_idx] = attn_out;
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
