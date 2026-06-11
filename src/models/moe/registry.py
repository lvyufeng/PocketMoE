from __future__ import annotations

from src.gguf.bundle import GGUFBundle
from src.models.moe.spec import MoEModelSpec


_SPECS: dict[str, MoEModelSpec] | None = None


def _init_specs() -> dict[str, MoEModelSpec]:
    from src.models.deepseek_v4.spec import DeepSeekV4Spec
    from src.models.minimax_m2.spec import MiniMaxM2Spec

    return {
        DeepSeekV4Spec.architecture: DeepSeekV4Spec(),
        MiniMaxM2Spec.architecture: MiniMaxM2Spec(),
    }


def _specs() -> dict[str, MoEModelSpec]:
    global _SPECS
    if _SPECS is None:
        _SPECS = _init_specs()
    return _SPECS


def known_architectures() -> list[str]:
    return sorted(_specs())


def get_spec(architecture: str) -> MoEModelSpec:
    key = architecture.strip().lower()
    try:
        return _specs()[key]
    except KeyError as exc:
        raise ValueError(f"unsupported MoE architecture {architecture!r}; known: {', '.join(known_architectures())}") from exc


def detect_spec(bundle: GGUFBundle, override: str = "auto") -> MoEModelSpec:
    if override != "auto":
        return get_spec(override)
    arch = bundle.metadata.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise ValueError("GGUF metadata does not contain general.architecture; pass --architecture explicitly")
    return get_spec(arch)
