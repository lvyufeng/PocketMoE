from pathlib import Path
import shutil

from setuptools import Extension, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "src" / "csrc"
EXTENSIONS_DIR = ROOT / "build" / "extensions"


class BuildExtensions(BuildExtension):
    def run(self):
        super().run()
        EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
        for ext in self.extensions:
            built_path = Path(self.get_ext_fullpath(ext.name)).resolve()
            if built_path.exists():
                shutil.copy2(built_path, EXTENSIONS_DIR / built_path.name)


setup(
    name="dsv4-inference-runtime",
    packages=[
        "src",
        "src.cli",
        "src.encoding",
        "src.gguf",
        "src.kernels",
        "src.models",
        "src.models.deepseek_v4",
        "src.models.minimax_m2",
        "src.models.moe",
        "src.runtime",
        "src.runtime.deepseek_v4",
        "src.runtime.moe",
        "src.server",
    ],
    ext_modules=[
        CUDAExtension(
            name="cuda_kernel",
            sources=[
                str(CSRC / "cuda_kernel.cpp"),
                str(CSRC / "cuda_kernel_impl.cu"),
            ],
            libraries=["cublas"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        ),
        Extension(
            name="deepseek_cpu_moe_ext",
            sources=[str(CSRC / "deepseek_cpu_moe_ext.cpp")],
            extra_compile_args=["-O3", "-mavx2", "-mfma", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
        CUDAExtension(
            name="moe_dispatch_cuda_ext",
            sources=[
                str(CSRC / "moe_dispatch_cuda_ext.cpp"),
                str(CSRC / "moe_dispatch_cuda_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtensions},
)
