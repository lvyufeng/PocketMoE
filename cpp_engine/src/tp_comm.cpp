#include "tp_comm.hpp"

#ifdef DSV4_HAVE_NCCL
#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>
#endif

namespace dsv4 {

bool nccl_available() {
#ifdef DSV4_HAVE_NCCL
    return true;
#else
    return false;
#endif
}

#ifdef DSV4_HAVE_NCCL
namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

void check_nccl(ncclResult_t err, const char* what) {
    if (err != ncclSuccess) throw std::runtime_error(std::string(what) + ": " + ncclGetErrorString(err));
}

ncclUniqueId load_or_create_id(int rank, const char* path) {
    ncclUniqueId id;
    if (rank == 0) {
        std::ifstream existing(path, std::ios::binary);
        if (existing) {
            existing.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (existing.gcount() == static_cast<std::streamsize>(sizeof(id))) return id;
        }
        check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to write NCCL id file");
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        if (!out) throw std::runtime_error("failed to write NCCL id bytes");
        return id;
    }
    for (int attempt = 0; attempt < 300; ++attempt) {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (in.gcount() == static_cast<std::streamsize>(sizeof(id))) return id;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("timed out waiting for NCCL id file");
}

struct CachedComm {
    ncclComm_t comm = nullptr;
};

ncclComm_t cached_comm(int world, int rank, int device, const char* id_path) {
    static std::unordered_map<std::string, CachedComm> comms;
    const std::string key = std::to_string(world) + ":" + std::to_string(rank) + ":" + std::to_string(device) + ":" + id_path;
    auto it = comms.find(key);
    if (it != comms.end()) return it->second.comm;
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    ncclUniqueId id = load_or_create_id(rank, id_path);
    ncclComm_t comm;
    check_nccl(ncclCommInitRank(&comm, world, id, rank), "ncclCommInitRank");
    comms.emplace(key, CachedComm{comm});
    return comm;
}

}  // namespace

void run_nccl_float_sum_smoke(int world, int rank, int device, const char* id_path, float value) {
    if (world <= 0 || rank < 0 || rank >= world) throw std::runtime_error("invalid NCCL world/rank");
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    ncclUniqueId id = load_or_create_id(rank, id_path);
    ncclComm_t comm;
    check_nccl(ncclCommInitRank(&comm, world, id, rank), "ncclCommInitRank");
    float* d_value = nullptr;
    float* d_sum = nullptr;
    check_cuda(cudaMalloc(&d_value, sizeof(float)), "cudaMalloc nccl value");
    check_cuda(cudaMalloc(&d_sum, sizeof(float)), "cudaMalloc nccl sum");
    check_cuda(cudaMemcpy(d_value, &value, sizeof(float), cudaMemcpyHostToDevice), "copy nccl input");
    check_nccl(ncclAllReduce(d_value, d_sum, 1, ncclFloat, ncclSum, comm, nullptr), "ncclAllReduce");
    check_cuda(cudaDeviceSynchronize(), "sync nccl smoke");
    float sum = 0.0f;
    check_cuda(cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost), "copy nccl sum");
    std::cout << "nccl_smoke rank=" << rank << " value=" << value << " sum=" << sum << "\n";
    cudaFree(d_value);
    cudaFree(d_sum);
    ncclCommDestroy(comm);
}

void nccl_all_reduce_sum_float_inplace(int world, int rank, int device, const char* id_path, float* d_values, int count) {
    if (world <= 0 || rank < 0 || rank >= world || d_values == nullptr || count <= 0) throw std::runtime_error("invalid NCCL all-reduce args");
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    check_nccl(ncclAllReduce(d_values, d_values, count, ncclFloat, ncclSum, comm, nullptr), "ncclAllReduce inplace");
}

void nccl_all_reduce_sum_bf16_inplace(int world, int rank, int device, const char* id_path, uint16_t* d_values, int count) {
    if (world <= 0 || rank < 0 || rank >= world || d_values == nullptr || count <= 0) throw std::runtime_error("invalid NCCL bf16 all-reduce args");
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    check_nccl(ncclAllReduce(d_values, d_values, count, ncclBfloat16, ncclSum, comm, nullptr), "ncclAllReduce bf16 inplace");
}

TpTopResult nccl_global_top1(int world, int rank, int device, const char* id_path, int local_token, float local_logit) {
    if (world <= 0 || rank < 0 || rank >= world) throw std::runtime_error("invalid NCCL world/rank");
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    float local[2] = {local_logit, static_cast<float>(local_token)};
    float* d_local = nullptr;
    float* d_all = nullptr;
    check_cuda(cudaMalloc(&d_local, 2 * sizeof(float)), "cudaMalloc local top");
    check_cuda(cudaMalloc(&d_all, static_cast<size_t>(world) * 2 * sizeof(float)), "cudaMalloc gathered top");
    check_cuda(cudaMemcpy(d_local, local, 2 * sizeof(float), cudaMemcpyHostToDevice), "copy local top");
    check_nccl(ncclAllGather(d_local, d_all, 2, ncclFloat, comm, nullptr), "ncclAllGather top");
    check_cuda(cudaDeviceSynchronize(), "sync top gather");
    std::vector<float> all(static_cast<size_t>(world) * 2);
    check_cuda(cudaMemcpy(all.data(), d_all, all.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy gathered top");
    TpTopResult result;
    result.logit = -INFINITY;
    for (int r = 0; r < world; ++r) {
        const float logit = all[static_cast<size_t>(r) * 2];
        const int token = static_cast<int>(all[static_cast<size_t>(r) * 2 + 1]);
        if (logit > result.logit) {
            result.logit = logit;
            result.token = token;
        }
    }
    cudaFree(d_local);
    cudaFree(d_all);
    return result;
}
#endif

}  // namespace dsv4
