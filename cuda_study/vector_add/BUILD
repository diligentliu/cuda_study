load("@rules_cuda//cuda:defs.bzl", "cuda_binary")

package(default_visibility = ["//visibility:public"])

cuda_binary(
    name = "vector_add",
    srcs = ["vector_add.cu"],
    deps = [
        "//cuda_study/util:util",
        "//cuda_study/util:perf_util",
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_glog//:glog",
    ],
)
