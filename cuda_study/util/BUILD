cc_library(
    name = "util",
    hdrs = ["util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_glog//:glog",
    ],
)

cc_library(
    name = "perf_util",
    hdrs = ["perf_util.h"],
    srcs = ["perf_util.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//cuda_study/util:util",
        "@com_github_gflags_gflags//:gflags",
        "@com_github_google_glog//:glog",
    ],
)