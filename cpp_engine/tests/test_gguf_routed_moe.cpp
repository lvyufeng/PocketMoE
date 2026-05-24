// Phase 3 step: GGUF multi-active-expert routed Q2 MoE smoke for layer 0.
// Embed -> ffn_norm -> tid2eid (hash gate) -> stage top-k experts ->
// batched IQ2_XXS w1/w3 -> SwiGLU + route_weight + Q8_1 quantize ->
// batched Q2_K w2 (atomicAdd across routes) -> MoE residual output.

#include "dsv4_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_routed_moe <model.gguf> [token]\n";
        return 2;
    }
    const int token = argc >= 3 ? std::atoi(argv[2]) : 1234;
    try {
        auto r = dsv4::run_gguf_routed_moe_smoke(argv[1], token);
        std::printf("token=%d dim=%d moe_inter_dim=%d n_active=%d\n",
                    token, r.dim, r.moe_inter_dim, r.n_active);
        std::printf("expert_ids   = [");
        for (int k = 0; k < r.n_active; ++k) {
            std::printf("%d%s", r.expert_ids[k], k + 1 < r.n_active ? ", " : "");
        }
        std::printf("]\n");
        std::printf("ffn_normed_rms = %.4f\n", r.ffn_normed_rms);
        std::printf("moe_out_rms    = %.4f\n", r.moe_out_rms);
        std::printf("moe_out[0..3]  = %.4f %.4f %.4f %.4f\n",
                    r.moe_out_first[0], r.moe_out_first[1],
                    r.moe_out_first[2], r.moe_out_first[3]);

        if (!(r.n_active >= 1 && r.n_active <= 8)) {
            std::cerr << "[FAIL] n_active range\n"; return 1;
        }
        // Distinct expert ids: hash-gate top-k typically returns no dupes.
        for (int i = 0; i < r.n_active; ++i) {
            for (int j = i + 1; j < r.n_active; ++j) {
                if (r.expert_ids[i] == r.expert_ids[j]) {
                    std::cerr << "[FAIL] duplicate expert id at slots "
                              << i << "," << j << "\n";
                    return 1;
                }
            }
            if (r.expert_ids[i] < 0 || r.expert_ids[i] >= 256) {
                std::cerr << "[FAIL] expert id out of range: "
                          << r.expert_ids[i] << "\n";
                return 1;
            }
        }
        if (!(r.ffn_normed_rms > 1e-4f && r.ffn_normed_rms < 1000.0f)) {
            std::cerr << "[FAIL] ffn_normed_rms out of range\n"; return 1;
        }
        if (!(r.moe_out_rms > 1e-6f && r.moe_out_rms < 1000.0f)) {
            std::cerr << "[FAIL] moe_out_rms out of range\n"; return 1;
        }
        std::cout << "[PASS] gguf routed multi-active MoE smoke\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
