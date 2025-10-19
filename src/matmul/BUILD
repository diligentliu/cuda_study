load("@rules_cuda//cuda:defs.bzl", "cuda_binary")

package(default_visibility = ["//visibility:public"])

cuda_binary(
    name = "matmul_basic",
    srcs = ["matmul_basic.cu"],
)

cuda_binary(
    name = "matmul_shared",
    srcs = ["matmul_shared.cu"],
)