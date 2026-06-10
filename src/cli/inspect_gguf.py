from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

from src.gguf.bundle import GGUFBundle, read_gguf_bundle
from src.gguf.ds4_mapping import validate_ds4_tensor_mappings
from src.gguf.reader import GGUFArraySummary
from src.moe_model.registry import detect_spec, known_architectures
from src.moe_model.spec import CapabilityReport, SpecValidation


DS4_Q2_TYPES = {
    "q2_k",
    "iq2_xxs",
    "iq1_m",
    "q4_k",
    "q8_0",
    "f16",
    "f32",
    "bf16",
    "i32",
}


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    unit = units[0]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{int(amount)} {unit}"
    return f"{amount:.2f} {unit}"


def _metadata_value_text(value) -> str:
    if isinstance(value, GGUFArraySummary):
        return f"array<{value.value_type_name}>[{value.length}]"
    return repr(value)


def _classify_tensor(name: str) -> str:
    if "ffn_gate_exps" in name or "ffn_up_exps" in name or "ffn_down_exps" in name:
        return "routed_experts"
    if "ffn_gate_shexp" in name or "ffn_up_shexp" in name or "ffn_down_shexp" in name:
        return "shared_experts"
    if ".attn" in name or "attn_" in name:
        return "attention"
    if "token_embd" in name or name.startswith("output") or ".output" in name:
        return "embedding_output"
    if "ffn" in name:
        return "ffn_other"
    return "other"


def _summarize_file(ds4) -> None:
    print(f"path: {ds4.path}")
    print(f"size: {_format_bytes(ds4.size)}")
    print(f"version: {ds4.version}")
    print(f"tensor_count: {ds4.tensor_count}")
    print(f"metadata_count: {ds4.metadata_count}")
    print(f"alignment: {ds4.alignment}")
    print(f"data_start: {ds4.data_start}")
    _summarize_metadata_and_tensors(ds4.metadata, ds4.tensors)


def _summarize_bundle(bundle: GGUFBundle) -> None:
    if len(bundle.shards) == 1:
        _summarize_file(bundle.primary)
        return
    print(f"path: {bundle.path}")
    print(f"size: {_format_bytes(bundle.size)}")
    print(f"version: {bundle.version}")
    print(f"shard_count: {len(bundle.shards)}")
    print(f"tensor_count: {bundle.tensor_count}")
    print(f"metadata_count: {bundle.metadata_count}")
    print(f"primary_shard: {bundle.shards[bundle.primary_index].path}")
    print("\nshards:")
    for shard in bundle.shards:
        print(
            f"  [{shard.index}] {os.path.basename(shard.path)} "
            f"size={_format_bytes(shard.file.size)} tensors={shard.file.tensor_count} metadata={shard.file.metadata_count}"
        )
    _summarize_metadata_and_tensors(bundle.metadata, bundle.tensors)


def _summarize_metadata_and_tensors(metadata, tensors) -> None:
    metadata_keys = (
        "general.architecture",
        "deepseek4.block_count",
        "deepseek4.embedding_length",
        "deepseek4.expert_count",
        "deepseek4.expert_used_count",
        "deepseek4.expert_feed_forward_length",
        "deepseek4.context_length",
        "minimax-m2.block_count",
        "minimax-m2.embedding_length",
        "minimax-m2.expert_count",
        "minimax-m2.expert_used_count",
        "minimax-m2.expert_feed_forward_length",
        "minimax-m2.context_length",
        "minimax-m2.attention.head_count",
        "minimax-m2.attention.head_count_kv",
        "minimax-m2.attention.key_length",
        "minimax-m2.attention.value_length",
    )
    for key in metadata_keys:
        if key in metadata:
            print(f"metadata.{key}: {_metadata_value_text(metadata[key])}")

    by_type = Counter(t.type_name for t in tensors)
    by_class = Counter(_classify_tensor(t.name) for t in tensors)
    bytes_by_type = defaultdict(int)
    bytes_by_class = defaultdict(int)
    for tensor in tensors:
        if tensor.nbytes is not None:
            bytes_by_type[tensor.type_name] += tensor.nbytes
            bytes_by_class[_classify_tensor(tensor.name)] += tensor.nbytes

    print("\ntensor types:")
    for type_name, count in sorted(by_type.items()):
        print(f"  {type_name:12s} {count:6d} {_format_bytes(bytes_by_type.get(type_name, 0))}")

    print("\ntensor classes:")
    for class_name, count in sorted(by_class.items()):
        print(f"  {class_name:18s} {count:6d} {_format_bytes(bytes_by_class.get(class_name, 0))}")

    routed = [t for t in tensors if _classify_tensor(t.name) == "routed_experts"]
    routed_types = Counter(t.type_name for t in routed)
    print("\nrouted expert types:")
    for type_name, count in sorted(routed_types.items()):
        print(f"  {type_name:12s} {count:6d}")

    unknown_types = sorted({t.type_name for t in tensors if t.type_name.startswith("unknown_")})
    if unknown_types:
        print("\nunknown tensor types:")
        for type_name in unknown_types:
            print(f"  {type_name}")


def _print_tensors(bundle: GGUFBundle, limit: int, contains: str | None) -> None:
    tensors = list(bundle.tensors)
    if contains:
        tensors = [t for t in tensors if contains in t.name]
    for tensor in tensors[:limit]:
        shard_text = "" if len(bundle.shards) == 1 else f"\tshard={os.path.basename(tensor.shard_path)}"
        print(
            f"{tensor.name}\t{tensor.type_name}\t{tensor.dimensions}\t"
            f"offset={tensor.offset}\tabs={tensor.absolute_offset}\tbytes={_format_bytes(tensor.nbytes)}{shard_text}"
        )
    if len(tensors) > limit:
        print(f"... {len(tensors) - limit} more tensors")


def _validate_ds4_q2(ds4) -> int:
    errors: list[str] = []
    arch = ds4.metadata.get("general.architecture")
    if arch != "deepseek4":
        errors.append(f"general.architecture expected 'deepseek4', got {arch!r}")
    if ds4.version != 3:
        errors.append(f"GGUF version expected 3, got {ds4.version}")

    names = ds4.tensors_by_name
    required_metadata = (
        "deepseek4.block_count",
        "deepseek4.embedding_length",
        "deepseek4.expert_count",
        "deepseek4.expert_used_count",
        "deepseek4.expert_feed_forward_length",
    )
    for key in required_metadata:
        if key not in ds4.metadata:
            errors.append(f"missing metadata {key}")

    try:
        n_layers = int(ds4.metadata.get("deepseek4.block_count", 43))
    except Exception:
        n_layers = 43
    for layer in range(n_layers):
        for suffix, expected_type in (
            ("ffn_gate_exps.weight", "iq2_xxs"),
            ("ffn_up_exps.weight", "iq2_xxs"),
            ("ffn_down_exps.weight", "q2_k"),
            ("ffn_gate_shexp.weight", "q8_0"),
            ("ffn_up_shexp.weight", "q8_0"),
            ("ffn_down_shexp.weight", "q8_0"),
        ):
            name = f"blk.{layer}.{suffix}"
            tensor = names.get(name)
            if tensor is None:
                errors.append(f"missing tensor {name}")
            elif tensor.type_name != expected_type:
                errors.append(f"{name} expected {expected_type}, got {tensor.type_name}")

    unsupported = sorted({t.type_name for t in ds4.tensors if t.type_name not in DS4_Q2_TYPES})
    if unsupported:
        errors.append("unsupported tensor types: " + ", ".join(unsupported))

    if errors:
        print("validation: FAILED")
        for error in errors[:50]:
            print(f"  - {error}")
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more errors")
        return 1
    print("validation: OK")
    return 0


def _runtime_state_shapes(config_path: str, routed_experts_device: str) -> dict[str, tuple[int, ...]]:
    import torch

    from src.runtime.transformer import ModelArgs, Transformer

    with open(config_path) as f:
        config_data = json.load(f)
    config_data["routed_experts_device"] = routed_experts_device
    config_data["max_batch_size"] = 1
    config_data["max_seq_len"] = 16
    if "nextn_predict_layers" in config_data:
        config_data["n_mtp_layers"] = int(config_data.pop("nextn_predict_layers"))
    if "n_mtp_layers" not in config_data:
        config_data["n_mtp_layers"] = 0
    with torch.device("meta"):
        model = Transformer(ModelArgs(**config_data))
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def _validate_runtime_mapping(ds4, config_path: str, routed_experts_device: str) -> int:
    validation = validate_ds4_tensor_mappings(
        ds4,
        _runtime_state_shapes(config_path, routed_experts_device),
    )
    print("\nruntime mapping:")
    print(f"  mappings: {len(validation.mappings)}")
    print(f"  missing_sources: {len(validation.missing_sources)}")
    print(f"  missing_targets: {len(validation.missing_targets)}")
    print(f"  shape_errors: {len(validation.shape_errors)}")
    print(f"  unmapped_sources: {len(validation.unmapped_sources)}")
    print(f"  unmapped_targets: {len(validation.unmapped_targets)}")
    errors = (
        validation.missing_sources
        + validation.missing_targets
        + validation.shape_errors
        + [f"unmapped GGUF tensor {name}" for name in validation.unmapped_sources]
        + [f"unmapped runtime tensor {name}" for name in validation.unmapped_targets]
    )
    if errors:
        print("runtime mapping: FAILED")
        for error in errors[:80]:
            print(f"  - {error}")
        if len(errors) > 80:
            print(f"  ... {len(errors) - 80} more errors")
        return 1
    print("runtime mapping: OK")
    return 0


def _print_spec_summary(report: CapabilityReport) -> None:
    p = report.params
    print("\nspec summary:")
    print(f"  architecture: {p.architecture}")
    print(f"  layers: {p.n_layers}")
    print(f"  hidden_size: {p.hidden_size}")
    print(f"  vocab_size: {p.vocab_size}")
    print(f"  context_length: {p.context_length}")
    print(f"  attention_kind: {p.attention_kind}")
    print(f"  n_heads: {p.n_heads}")
    print(f"  n_kv_heads: {p.n_kv_heads}")
    print(f"  head_dim: {p.head_dim}")
    print(f"  rope_dim: {p.rope_dim}")
    print(f"  rope_base: {p.rope_base}")
    print(f"  n_routed_experts: {p.n_routed_experts}")
    print(f"  top_k: {p.top_k}")
    print(f"  expert_intermediate_size: {p.expert_intermediate_size}")
    print(f"  n_shared_experts: {p.n_shared_experts}")
    print(f"  gate_function: {p.gate_function}")


def _print_validation(validation: SpecValidation) -> int:
    print("\nspec validation:")
    print(f"  ok: {validation.ok}")
    print(f"  mapped_sources: {validation.mapped_sources}")
    print(f"  unmapped_sources: {len(validation.unmapped_sources)}")
    print("  role_counts:")
    for role, count in sorted(validation.role_counts.items()):
        print(f"    {role:16s} {count:6d}")
    if validation.warnings:
        print("  warnings:")
        for warning in validation.warnings[:20]:
            print(f"    - {warning}")
    if validation.errors:
        print("  errors:")
        for error in validation.errors[:80]:
            print(f"    - {error}")
        if len(validation.errors) > 80:
            print(f"    ... {len(validation.errors) - 80} more errors")
        return 1
    return 0


def _print_capability_report(report: CapabilityReport) -> None:
    print("\ncapability report:")
    print("  tensor types:")
    for type_name, count in sorted(report.tensor_type_counts.items()):
        print(f"    {type_name:12s} {count:6d} {_format_bytes(report.bytes_by_type.get(type_name, 0))}")
    print("  tensor roles:")
    for role, count in sorted(report.tensor_role_counts.items()):
        print(f"    {role:16s} {count:6d} {_format_bytes(report.bytes_by_role.get(role, 0))}")
    print("  capabilities:")
    for item in report.capabilities:
        print(f"    [{item.status}] {item.name}: {item.reason}")


def _print_placement_report(report: CapabilityReport) -> None:
    print("\nplacement report:")
    for decision in report.placements:
        est = ""
        if decision.estimated_bytes is not None:
            est += f" total={_format_bytes(decision.estimated_bytes)}"
        if decision.estimated_bytes_per_gpu is not None:
            est += f" per_gpu={_format_bytes(decision.estimated_bytes_per_gpu)}"
        print(f"  [{decision.status}] {decision.name}:{est} {decision.reason}")


def _print_moe_runtime_report(bundle: GGUFBundle, *, gpu_count: int, gpu_memory_gib: float):
    from src.runtime.minimax_m2_moe import build_minimax_m2_moe_runtime_plan

    plan = build_minimax_m2_moe_runtime_plan(bundle, gpu_count=gpu_count, gpu_memory_gib=gpu_memory_gib)
    print("\nmoe runtime report:")
    print(f"  architecture: {plan.architecture}")
    print(f"  minimax_moe_runtime: {plan.status}")
    print(f"  ok: {plan.ok}")
    print(f"  layers: {plan.params.n_layers}")
    print(f"  routed tensors: {plan.routed_tensor_count} / expected {plan.expected_routed_tensor_count}")
    print(f"  moe tensors: {plan.moe_tensor_count} / expected {plan.expected_moe_tensor_count}")
    print(f"  routed bytes: {_format_bytes(plan.routed_bytes)}")
    print(f"  moe bytes: {_format_bytes(plan.moe_bytes)}")
    print("  role counts:")
    for role, count in sorted(plan.tensor_role_counts.items()):
        print(f"    {role:16s} {count:6d} {_format_bytes(plan.bytes_by_role.get(role, 0))}")
    print("  routed types:")
    for type_name, count in sorted(plan.routed_type_counts.items()):
        print(f"    {type_name:12s} {count:6d}")
    print("  skipped/deferred dense components:")
    skipped_by_role = Counter((item.role, item.type_name, item.reason) for item in plan.skipped_tensors)
    for (role, type_name, reason), count in sorted(skipped_by_role.items()):
        print(f"    [{type_name}] {role:16s} {count:6d}: {reason}")
    print("  placements:")
    for decision in plan.placements:
        est = ""
        if decision.estimated_bytes is not None:
            est += f" total={_format_bytes(decision.estimated_bytes)}"
        if decision.estimated_bytes_per_gpu is not None:
            est += f" per_gpu={_format_bytes(decision.estimated_bytes_per_gpu)}"
        print(f"    [{decision.status}] {decision.name}:{est} {decision.reason}")
    print("  generation: deferred (MiniMax-M2 full attention/q4/q5 runtime is not implemented yet)")
    if plan.warnings:
        print("  warnings:")
        for warning in plan.warnings[:20]:
            print(f"    - {warning}")
    if plan.errors:
        print("  errors:")
        for error in plan.errors[:80]:
            print(f"    - {error}")
    return plan


def _check_minimax_routed_blocks(bundle: GGUFBundle, *, layer_limit: int, expert: int, row_count: int) -> int:
    from src.runtime.minimax_m2_moe import MiniMaxM2RoutedBlockLoader, ROUTED_ROLES, build_minimax_m2_moe_runtime_plan

    plan = build_minimax_m2_moe_runtime_plan(bundle)
    if not plan.ok:
        print("\nrouted block check: FAILED")
        for error in plan.errors[:20]:
            print(f"  - {error}")
        return 1
    layers = min(max(int(layer_limit), 0), plan.params.n_layers)
    rows = max(int(row_count), 1)
    print("\nrouted block check:")
    print(f"  layers_checked: {layers}")
    print(f"  expert: {expert}")
    print(f"  row_count: {rows}")
    with MiniMaxM2RoutedBlockLoader(bundle, plan=plan) as loader:
        for layer in range(layers):
            for role in sorted(ROUTED_ROLES):
                blocks, type_name, in_dim = loader.read_expert_role_blocks(layer, role, expert=int(expert), row_count=rows)
                print(f"  layer={layer} role={role} type={type_name} in_dim={in_dim} blocks_shape={tuple(blocks.shape)}")
    print("routed block check: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a GGUF file or sharded GGUF bundle without loading tensor payloads.")
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--list-tensors", action="store_true")
    parser.add_argument("--contains", default=None, help="Only list tensors containing this substring")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--validate-ds4-q2", action="store_true")
    parser.add_argument("--validate-runtime-mapping", action="store_true")
    parser.add_argument("--config", default="configs/config_w8a8.json")
    parser.add_argument("--routed-experts-device", choices=["gpu", "cpu"], default="cpu")
    parser.add_argument("--architecture", default="auto", choices=["auto", *known_architectures()])
    parser.add_argument("--spec-summary", action="store_true")
    parser.add_argument("--validate-spec", action="store_true")
    parser.add_argument("--capability-report", action="store_true")
    parser.add_argument("--placement-report", action="store_true")
    parser.add_argument("--moe-runtime-report", action="store_true", help="Report MiniMax-M2 MoE-only runtime readiness")
    parser.add_argument("--check-routed-blocks", action="store_true", help="Read a tiny MiniMax-M2 routed expert block slice to validate payload offsets")
    parser.add_argument("--check-layer-limit", type=int, default=1, help="Number of layers to sample for --check-routed-blocks")
    parser.add_argument("--check-expert", type=int, default=0, help="Expert id to sample for --check-routed-blocks")
    parser.add_argument("--check-row-count", type=int, default=1, help="Output rows to sample for --check-routed-blocks")
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--gpu-memory-gib", type=float, default=22.0)
    args = parser.parse_args()

    if not os.path.exists(args.gguf_path):
        raise FileNotFoundError(args.gguf_path)

    bundle = read_gguf_bundle(args.gguf_path)
    if args.summary or not (
        args.list_tensors
        or args.spec_summary
        or args.validate_spec
        or args.capability_report
        or args.placement_report
        or args.moe_runtime_report
        or args.check_routed_blocks
    ):
        _summarize_bundle(bundle)
    if args.list_tensors:
        _print_tensors(bundle, args.limit, args.contains)

    status = 0
    if args.validate_ds4_q2:
        status = max(status, _validate_ds4_q2(bundle if len(bundle.shards) > 1 else bundle.primary))

    spec = None
    report = None
    if (
        args.spec_summary
        or args.validate_spec
        or args.capability_report
        or args.placement_report
        or args.validate_runtime_mapping
        or args.moe_runtime_report
        or args.check_routed_blocks
    ):
        spec = detect_spec(bundle, args.architecture)

    if args.validate_runtime_mapping:
        assert spec is not None
        if spec.architecture != "deepseek4":
            print(
                "runtime mapping: FAILED\n"
                f"  - runtime mapping/generation is currently implemented for deepseek4 only, got {spec.architecture!r}; "
                "use --validate-spec for header/logical tensor validation or --moe-runtime-report for MiniMax-M2 MoE-only readiness",
                flush=True,
            )
            status = max(status, 1)
        else:
            status = max(status, _validate_runtime_mapping(bundle if len(bundle.shards) > 1 else bundle.primary, args.config, args.routed_experts_device))

    if spec is not None and (args.spec_summary or args.capability_report or args.placement_report):
        report = spec.capability_report(bundle, gpu_count=args.gpu_count, gpu_memory_gib=args.gpu_memory_gib)
    if args.spec_summary:
        assert report is not None
        _print_spec_summary(report)
    if args.validate_spec:
        assert spec is not None
        status = max(status, _print_validation(spec.validate_bundle(bundle)))
    if args.capability_report:
        assert report is not None
        _print_capability_report(report)
    if args.placement_report:
        assert report is not None
        _print_placement_report(report)
    if args.moe_runtime_report or args.check_routed_blocks:
        assert spec is not None
        if spec.architecture != "minimax-m2":
            print(
                "\nmoe runtime report: FAILED\n"
                f"  - MoE runtime readiness report is currently implemented for minimax-m2 only, got {spec.architecture!r}",
                flush=True,
            )
            status = max(status, 1)
        else:
            plan = _print_moe_runtime_report(bundle, gpu_count=args.gpu_count, gpu_memory_gib=args.gpu_memory_gib)
            if not plan.ok:
                status = max(status, 1)
            if args.check_routed_blocks:
                status = max(status, _check_minimax_routed_blocks(
                    bundle,
                    layer_limit=args.check_layer_limit,
                    expert=args.check_expert,
                    row_count=args.check_row_count,
                ))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
