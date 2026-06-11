from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

REMOVED_IMPORT_PREFIXES = (
    "src.gguf",
    "src.models.moe",
    "src.runtime.deepseek_v4",
    "src.runtime.moe",
    "src.moe",
    "src.moe_model",
)

RUNTIME_FORBIDDEN_IMPORT_PREFIXES = (
    "safetensors",
    "src.loader",
    "src.models",
)

LOADER_FORBIDDEN_IMPORT_PREFIXES = (
    "src.models",
    "src.runtime",
)

COMPONENTS_MOE_ALLOWED_MODEL_IMPORTS = {
    "src.models.deepseek_v4.spec",
    "src.models.minimax_m2.spec",
}

COMPONENTS_MOE_FORBIDDEN_MODEL_MODULES = (
    ".generation",
    ".loader",
    ".moe_runtime",
    ".moe_server",
    ".partition",
    ".runtime",
)


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _starts_with_any(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in prefixes)


def _format_violations(violations: list[tuple[Path, str]]) -> str:
    return "\n".join(f"{path.relative_to(REPO_ROOT)} imports {module}" for path, module in violations)


def test_removed_namespaces_are_not_imported_from_source() -> None:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(SRC_ROOT):
        for module in _imported_modules(path):
            if _starts_with_any(module, REMOVED_IMPORT_PREFIXES):
                violations.append((path, module))

    assert not violations, _format_violations(violations)


def test_runtime_stays_model_and_checkpoint_format_agnostic() -> None:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(SRC_ROOT / "runtime"):
        for module in _imported_modules(path):
            if _starts_with_any(module, RUNTIME_FORBIDDEN_IMPORT_PREFIXES):
                violations.append((path, module))

    assert not violations, _format_violations(violations)


def test_loader_does_not_depend_on_runtime_or_model_packages() -> None:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(SRC_ROOT / "loader"):
        for module in _imported_modules(path):
            if _starts_with_any(module, LOADER_FORBIDDEN_IMPORT_PREFIXES):
                violations.append((path, module))

    assert not violations, _format_violations(violations)


def test_components_moe_only_imports_model_specs_for_registry_discovery() -> None:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(SRC_ROOT / "components" / "moe"):
        for module in _imported_modules(path):
            if not _starts_with_any(module, ("src.models",)):
                continue
            if path.name == "registry.py" and module in COMPONENTS_MOE_ALLOWED_MODEL_IMPORTS:
                continue
            violations.append((path, module))

    assert not violations, _format_violations(violations)


def test_components_moe_does_not_import_model_runtime_loader_or_servers() -> None:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(SRC_ROOT / "components" / "moe"):
        for module in _imported_modules(path):
            if module.startswith("src.models") and module.endswith(COMPONENTS_MOE_FORBIDDEN_MODEL_MODULES):
                violations.append((path, module))

    assert not violations, _format_violations(violations)
