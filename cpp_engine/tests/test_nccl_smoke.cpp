#include "tp_comm.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    int world = 1;
    int rank = 0;
    int device = 0;
    std::string id_path = "/tmp/dsv4_nccl_id.bin";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--world" && i + 1 < argc) world = std::stoi(argv[++i]);
        else if (arg == "--rank" && i + 1 < argc) rank = std::stoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device = std::stoi(argv[++i]);
        else if (arg == "--id-path" && i + 1 < argc) id_path = argv[++i];
        else throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
    if (!dsv4::nccl_available()) {
        std::cout << "[SKIP] NCCL not available\n";
        return 0;
    }
#ifdef DSV4_HAVE_NCCL
    dsv4::run_nccl_float_sum_smoke(world, rank, device, id_path.c_str(), static_cast<float>(rank + 1));
    return 0;
#else
    return 0;
#endif
}
