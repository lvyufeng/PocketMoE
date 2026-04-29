from pathlib import Path
import ctypes
import importlib.util
from typing import Optional
import importlib.machinery

_EXT_PATH = Path(__file__).with_name("wo_a_cuda_ext.so")
_NATIVE_MOD = None
_PRELOADED_TORCH_LIBS = False


def _find_built_extension() -> Optional[Path]:
    if _EXT_PATH.exists():
        return _EXT_PATH
    suffixes = sorted(importlib.machinery.EXTENSION_SUFFIXES, key=len, reverse=True)
    for suffix in suffixes:
        candidate = Path(__file__).with_name(f"wo_a_cuda_ext{suffix}")
        if candidate.exists():
            return candidate
    return None


def _preload_torch_libs() -> None:
    global _PRELOADED_TORCH_LIBS
    if _PRELOADED_TORCH_LIBS:
        return
    import torch

    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    lib_names = [
        "libc10.so",
        "libc10_cuda.so",
        "libtorch_cpu.so",
        "libtorch_cuda.so",
        "libtorch_python.so",
        "libtorch.so",
    ]
    for name in lib_names:
        path = torch_lib_dir / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _PRELOADED_TORCH_LIBS = True


def load_wo_a_cuda_ext() -> Optional[object]:
    global _NATIVE_MOD
    if _NATIVE_MOD is not None:
        return _NATIVE_MOD
    ext_path = _find_built_extension()
    if ext_path is None:
        return None
    try:
        _preload_torch_libs()
        spec = importlib.util.spec_from_file_location("wo_a_cuda_ext", ext_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _NATIVE_MOD = module
        return module
    except Exception:
        return None
