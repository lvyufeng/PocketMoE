from pathlib import Path
import ctypes
import importlib.machinery
import importlib.util
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXT_DIR = _REPO_ROOT / "build" / "extensions"
_NATIVE_MOD = None
_PRELOADED_TORCH_LIBS = False


def _find_built_extension(module_name: str) -> Optional[Path]:
    suffixes = sorted(importlib.machinery.EXTENSION_SUFFIXES, key=len, reverse=True)
    candidates = [f"{module_name}.so"] + [f"{module_name}{suffix}" for suffix in suffixes]
    for name in candidates:
        path = _EXT_DIR / name
        if path.exists():
            return path
    matches = sorted(_EXT_DIR.glob(f"{module_name}*.so"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _preload_torch_libs() -> None:
    global _PRELOADED_TORCH_LIBS
    if _PRELOADED_TORCH_LIBS:
        return
    import torch

    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    for name in ("libc10.so", "libc10_cuda.so", "libtorch_cpu.so", "libtorch_cuda.so", "libtorch_python.so", "libtorch.so"):
        path = torch_lib_dir / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _PRELOADED_TORCH_LIBS = True


def _load_extension(module_name: str) -> Optional[object]:
    ext_path = _find_built_extension(module_name)
    if ext_path is None:
        return None
    try:
        _preload_torch_libs()
        spec = importlib.util.spec_from_file_location(module_name, ext_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def load_moe_dispatch_cuda_ext() -> Optional[object]:
    global _NATIVE_MOD
    if _NATIVE_MOD is None:
        _NATIVE_MOD = _load_extension("moe_dispatch_cuda_ext")
    return _NATIVE_MOD
