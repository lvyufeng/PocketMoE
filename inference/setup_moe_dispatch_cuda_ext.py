import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

setup(
    name="moe_dispatch_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="moe_dispatch_cuda_ext",
            sources=[
                str(ROOT / "moe_dispatch_cuda_ext.cpp"),
                str(ROOT / "moe_dispatch_cuda_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
