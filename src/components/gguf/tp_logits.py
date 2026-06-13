from __future__ import annotations

import torch
import torch.distributed as dist


def tp_vocab_row_range(total_rows: int, world: int, rank: int) -> tuple[int, int]:
    total_rows = int(total_rows)
    world = int(world)
    rank = int(rank)
    if total_rows <= 0:
        raise ValueError(f"total_rows must be positive, got {total_rows}")
    if world <= 0:
        raise ValueError(f"world must be positive, got {world}")
    if rank < 0 or rank >= world:
        raise ValueError(f"rank {rank} outside [0, {world})")
    start = (total_rows * rank) // world
    end = (total_rows * (rank + 1)) // world
    return start, end - start


def distributed_argmax_local_logits(logits: torch.Tensor, *, row_start: int = 0) -> torch.Tensor:
    local_values, local_indices = torch.max(logits, dim=-1)
    local_indices = local_indices.to(torch.long) + int(row_start)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        world = dist.get_world_size()
        all_values = [torch.empty_like(local_values) for _ in range(world)]
        all_indices = [torch.empty_like(local_indices) for _ in range(world)]
        dist.all_gather(all_values, local_values.contiguous())
        dist.all_gather(all_indices, local_indices.contiguous())
        values = torch.stack(all_values, dim=0)
        indices = torch.stack(all_indices, dim=0)
        best_rank = torch.argmax(values, dim=0)
        return indices.gather(0, best_rank.unsqueeze(0)).squeeze(0)
    return local_indices


def gather_sharded_logits(logits: torch.Tensor, *, full_out_dim: int, row_start: int = 0) -> torch.Tensor:
    if int(logits.size(-1)) == int(full_out_dim) and int(row_start) == 0:
        return logits
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return logits
    world = dist.get_world_size()
    local_n = torch.tensor([int(logits.size(-1))], device=logits.device, dtype=torch.long)
    all_n = [torch.empty_like(local_n) for _ in range(world)]
    dist.all_gather(all_n, local_n)
    sizes = [int(item.item()) for item in all_n]
    max_n = max(sizes)
    if int(logits.size(-1)) < max_n:
        pad = torch.empty((*logits.shape[:-1], max_n - int(logits.size(-1))), device=logits.device, dtype=logits.dtype)
        logits_send = torch.cat((logits, pad), dim=-1).contiguous()
    else:
        logits_send = logits.contiguous()
    gathered = [torch.empty_like(logits_send) for _ in range(world)]
    dist.all_gather(gathered, logits_send)
    parts = [part[..., :sizes[i]] for i, part in enumerate(gathered)]
    return torch.cat(parts, dim=-1)
