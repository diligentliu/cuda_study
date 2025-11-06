load("@rules_cuda//cuda:defs.bzl", "cuda_binary")

package(default_visibility = ["//visibility:public"])

cuda_binary(
    name = "matmul_basic",
    srcs = ["matmul_basic.cu"],
    deps = [
        "//cuda_study/util:util",
        "//cuda_study/util:perf_util",
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_glog//:glog",
    ],
)

cuda_binary(
    name = "matmul_shared",
    srcs = ["matmul_shared.cu"],
    deps = [
        "//cuda_study/util:util",
        "//cuda_study/util:perf_util",
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_glog//:glog",
    ],
)

cuda_binary(
    name = "matmul_tiled",
    srcs = ["matmul_tiled.cu"],
    deps = [
        "//cuda_study/util:util",
        "//cuda_study/util:perf_util",
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_glog//:glog",
    ],
)