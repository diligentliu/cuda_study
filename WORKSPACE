workspace(name = "cuda_study")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
    name = "rules_cuda",
    sha256 = "fe8d3d8ed52b9b433f89021b03e3c428a82e10ed90c72808cc4988d1f4b9d1b3",
    strip_prefix = "rules_cuda-v0.2.5",
    urls = ["https://github.com/bazel-contrib/rules_cuda/releases/download/v0.2.5/rules_cuda-v0.2.5.tar.gz"],
)

load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")
rules_cuda_dependencies()
register_detected_cuda_toolchains()

git_repository(
    name = "com_github_gflags_gflags",
    remote = "git@coding.jd.com:liusongkuo1/gflags.git",
    tag = "v2.2.2"
)

git_repository(
    name = "com_github_google_glog",
    remote = "git@coding.jd.com:liusongkuo1/glog.git",
    tag = "v0.7.1"
)

git_repository(
    name = "com_google_googletest",
    remote = "git@coding.jd.com:liusongkuo1/googletest.git",
    tag = "v1.17.0"
)
