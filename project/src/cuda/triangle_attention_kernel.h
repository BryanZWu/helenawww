namespace triangle_attention_gpu_ops
{
    struct TriangleAttentionDescriptor
    {
        int batch_size;
        int seq_len;
        int head_dim;
        int num_heads;
    };
}