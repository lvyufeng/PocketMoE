import os
from pathlib import Path

from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

setup(
    name="deepseek_cpu_moe_ext",
    ext_modules=[
        Extension(
            name="deepseek_cpu_moe_ext",
            sources=[str(ROOT / "deepseek_cpu_moe_ext.cpp")],
            extra_compile_args=["-O3", "-mavx2", "-mfma", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ],
)
