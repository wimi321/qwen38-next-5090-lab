from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from setuptools import Extension, setup


if sys.platform == "linux" and not os.environ.get("CUDA_HOME"):
    for _cuda_candidate in (Path("/usr/local/cuda"), Path("/usr/local/cuda-13.3")):
        if (_cuda_candidate / "bin" / "nvcc").is_file():
            os.environ["CUDA_HOME"] = str(_cuda_candidate)
            os.environ.setdefault("CUDA_PATH", str(_cuda_candidate))
            os.environ.setdefault("CUDACXX", str(_cuda_candidate / "bin" / "nvcc"))
            os.environ["PATH"] = (
                str(_cuda_candidate / "bin")
                + os.pathsep
                + os.environ.get("PATH", os.defpath)
            )
            break

from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension


ROOT = Path(__file__).parent
SOURCE_ROOT = ROOT.resolve()

# Release wheels are built from throw-away ext4 clones.  Strip absolute build
# paths and debug tables from the three native extensions so the binary does
# not disclose the builder's home directory and repeated builds have stable
# source locations in diagnostics.
RELEASE_CXX_FLAGS = [
    "-O3",
    "-std=c++17",
    "-g0",
    f"-ffile-prefix-map={SOURCE_ROOT}=.",
    f"-fdebug-prefix-map={SOURCE_ROOT}=.",
    f"-fmacro-prefix-map={SOURCE_ROOT}=.",
]


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _cuda_runtime_paths() -> tuple[list[str], list[str]]:
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    return [str(cuda_home / "include")], library_dirs


cuda_include_dirs, cuda_library_dirs = _cuda_runtime_paths()
_check_toolchain()

linux_ple_extensions = (
    [
        Extension(
            name="freetoken.models.qwen4_exp._ple_io_uring",
            sources=[
                "python/freetoken/models/qwen4_exp/csrc/ple_io_uring.cpp",
            ],
            extra_compile_args=[*RELEASE_CXX_FLAGS, "-pthread"],
            extra_link_args=["-pthread"],
            language="c++",
        )
    ]
    if sys.platform == "linux"
    else []
)


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=RELEASE_CXX_FLAGS,
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links cudart for the
        # cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        CppExtension(
            name="freetoken.kernel._cpu_moe",
            sources=[
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=[*RELEASE_CXX_FLAGS, "-pthread"],
        ),
        *linux_ple_extensions,
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
