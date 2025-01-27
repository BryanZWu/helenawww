/*
 * cpp wrapper around the triangle attention kernel.
 * It is used to register the kernel with pybind11.
 */
#include "triangle_attention_kernels.h"
#include "pybind11_kernel_helpers.h"

namespace {
pybind11::dict TriangleAttentionRegistrations() {
  pybind11::dict dict;
  dict["triangle_attention_kernel"] =
      gpu_ops::EncapsulateFunction(gpu_ops::triangle_attention_kernel);
  dict["triangle_attention_kernel_backward"] =
      gpu_ops::EncapsulateFunction(gpu_ops::triangle_attention_kernel_backward);
  return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
  m.def("get_triangle_attention_registrations", &TriangleAttentionRegistrations);
  m.def("create_triangle_attention_descriptor",
        [](int batch_size, int seq_len, int head_dim, int num_heads, bool from_starting_node) {
          return gpu_ops::PackDescriptor(gpu_ops::TriangleAttentionDescriptor{
              batch_size, seq_len, head_dim, num_heads, from_starting_node});
        });
}
} // namespace