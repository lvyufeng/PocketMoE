import os
import struct
import time
from multiprocessing import resource_tracker, shared_memory

import torch

_HEADER_FORMAT = "qqqq"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)
_MAX_ACK_RANKS = int(os.getenv("DEEPSEEK_CPU_MOE_MAX_ACK_RANKS", "8"))
_OUTPUT_SLOTS = int(os.getenv("DEEPSEEK_CPU_MOE_OUTPUT_SLOTS", "64"))


class CPUMoESharedMemory:
    def __init__(self, name: str, dim: int, topk: int, create: bool = False):
        self.name = name
        self.dim = dim
        self.topk = topk
        self.input_bytes = dim * 4
        self.ids_bytes = topk * 8
        self.weights_bytes = topk * 4
        self.output_bytes = dim * 4
        self.output_slots = _OUTPUT_SLOTS
        self.ack_bytes = self.output_slots * _MAX_ACK_RANKS * 8
        self.size = _HEADER_SIZE + self.input_bytes + self.ids_bytes + self.weights_bytes + self.output_bytes * self.output_slots + self.ack_bytes
        if create:
            try:
                stale = shared_memory.SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
        self.shm = shared_memory.SharedMemory(name=name, create=create, size=self.size if create else 0)
        if not create:
            try:
                resource_tracker.unregister(self.shm._name, "shared_memory")
            except Exception:
                pass
        self.buf = self.shm.buf
        self.input_offset = _HEADER_SIZE
        self.ids_offset = self.input_offset + self.input_bytes
        self.weights_offset = self.ids_offset + self.ids_bytes
        self.output_offset = self.weights_offset + self.weights_bytes
        self.ack_offset = self.output_offset + self.output_bytes * self.output_slots
        if create:
            self.write_header(0, 0, -1, 0)
            self.ack_tensor().zero_()

    def close(self, unlink: bool = False):
        self.shm.close()
        if unlink:
            self.shm.unlink()

    def read_header(self) -> tuple[int, int, int, int]:
        return struct.unpack_from(_HEADER_FORMAT, self.buf, 0)

    def write_header(self, request_seq: int, response_seq: int, layer_id: int, stop: int) -> None:
        struct.pack_into(_HEADER_FORMAT, self.buf, 0, request_seq, response_seq, layer_id, stop)

    def input_tensor(self) -> torch.Tensor:
        return torch.frombuffer(self.buf, dtype=torch.float32, count=self.dim, offset=self.input_offset).view(1, self.dim)

    def ids_tensor(self) -> torch.Tensor:
        return torch.frombuffer(self.buf, dtype=torch.long, count=self.topk, offset=self.ids_offset).view(1, self.topk)

    def weights_tensor(self) -> torch.Tensor:
        return torch.frombuffer(self.buf, dtype=torch.float32, count=self.topk, offset=self.weights_offset).view(1, self.topk)

    def output_tensor(self, seq: int | None = None) -> torch.Tensor:
        slot = 0 if seq is None else int(seq) % self.output_slots
        return torch.frombuffer(self.buf, dtype=torch.float32, count=self.dim, offset=self.output_offset + slot * self.output_bytes).view(1, self.dim)

    def ack_tensor(self) -> torch.Tensor:
        return torch.frombuffer(self.buf, dtype=torch.long, count=self.output_slots * _MAX_ACK_RANKS, offset=self.ack_offset).view(self.output_slots, _MAX_ACK_RANKS)

    def ack(self, rank: int, seq: int) -> None:
        slot = int(seq) % self.output_slots
        struct.pack_into("q", self.buf, self.ack_offset + (slot * _MAX_ACK_RANKS + int(rank)) * 8, int(seq))

    def wait_slot_acks(self, seq: int, world_size: int) -> None:
        if seq <= 0:
            return
        slot = int(seq) % self.output_slots
        while True:
            ok = True
            for rank in range(1, int(world_size)):
                if struct.unpack_from("q", self.buf, self.ack_offset + (slot * _MAX_ACK_RANKS + rank) * 8)[0] < seq:
                    ok = False
                    break
            if ok:
                return
            time.sleep(0)

    def wait_acks(self, seq: int, world_size: int) -> None:
        self.wait_slot_acks(seq, world_size)

    def submit(self, layer_id: int, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> int:
        req, resp, _layer, _stop = self.read_header()
        seq = req + 1
        self.input_tensor().copy_(x.detach().to(device="cpu", dtype=torch.float32).view(1, self.dim), non_blocking=False)
        self.ids_tensor().copy_(ids.detach().to(device="cpu", dtype=torch.long).view(1, self.topk), non_blocking=False)
        self.weights_tensor().copy_(weights.detach().to(device="cpu", dtype=torch.float32).view(1, self.topk), non_blocking=False)
        struct.pack_into("qqq", self.buf, 8, resp, layer_id, 0)
        struct.pack_into("q", self.buf, 0, seq)
        return seq

    def wait_response(self, seq: int) -> torch.Tensor:
        while True:
            _req, resp, _layer, stop = self.read_header()
            if stop:
                raise RuntimeError("CPU MoE server stopped")
            if resp >= seq:
                return self.output_tensor(seq)
            time.sleep(0)

    def wait_request(self, last_seq: int) -> tuple[int, int] | None:
        while True:
            req, _resp, layer_id, stop = self.read_header()
            if stop:
                return None
            if req > last_seq:
                return req, layer_id
            time.sleep(0)

    def respond(self, seq: int) -> None:
        req, _resp, layer_id, stop = self.read_header()
        struct.pack_into("q", self.buf, 0, req)
        struct.pack_into("q", self.buf, 16, layer_id)
        struct.pack_into("q", self.buf, 24, stop)
        struct.pack_into("q", self.buf, 8, seq)
