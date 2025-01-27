#include "triangle_attention.h"
#include <pybind11/pybind11.h>
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"

namespace
{

    void triangle_attention_cuda(void *out, const void **in, const char *opaque,
                                 size_t opaque_len, XlaCustomCallStatus *status)
    {
        const TriangleAttentionDescriptor *desc =
            reinterpret_cast<const TriangleAttentionDescriptor *>(opaque);

        const float *q = reinterpret_cast<const float *>(in[0]);
        const float *k = reinterpret_cast<const float *>(in[1]);
        const float *v = reinterpret_cast<const float *>(in[2]);
        const float *mask = reinterpret_cast<const float *>(in[3]);
        float *output = reinterpret_cast<float *>(out);

        cudaStream_t stream = nullptr; // TODO: Get the current stream from XLA

        cudaError_t error = LaunchTriangleAttention(
            const_cast<float *>(q),
            const_cast<float *>(k),
            const_cast<float *>(v),
            const_cast<float *>(mask),
            output,
            desc,
            stream);

        if (error != cudaSuccess)
        {
            XlaCustomCallStatusSetFailure(status, "Triangle attention kernel launch failed", 0);
        }
    }

} // namespace

REGISTER_CUSTOM_CALL_TARGET(triangle_attention_cuda, "CUDA");
