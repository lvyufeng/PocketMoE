// Phase 3 step: end-to-end greedy decode smoke. Loads the model once and runs
// a per-step forward over [seed_tokens] + [max_new_tokens] positions, with
// KV cache sized to the full sequence. Validates that:
//   - the run completes without OOM/throw,
//   - each step produces a valid in-range token,
//   - per-step wall time is reported for a coarse decode-tps signal.

#include "dsv4_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_generate <model.gguf> [max_new_tokens] [seed1 seed2 ...]\n";
        return 2;
    }
    const std::string ckpt = argv[1];
    const int max_new = argc >= 3 ? std::atoi(argv[2]) : 4;
    std::vector<int> seeds;
    for (int i = 3; i < argc; ++i) seeds.push_back(std::atoi(argv[i]));
    if (seeds.empty()) seeds = {1234};

    try {
        auto r = dsv4::run_gguf_generate_smoke(ckpt, seeds, max_new);
        std::printf("n_layers=%d dim=%d vocab=%d prompt_tokens=%d decode_tokens=%d\n",
                    r.n_layers, r.dim, r.vocab, r.prompt_tokens, r.decode_tokens);
        std::printf("load_seconds   = %.3f\n", r.load_seconds);
        std::printf("forward_seconds= %.3f (%d positions; %.1f ms/step)\n",
                    r.forward_seconds,
                    static_cast<int>(r.top_logits.size()),
                    r.top_logits.empty() ? 0.0
                                         : 1000.0 * r.forward_seconds /
                                               static_cast<double>(r.top_logits.size()));
        if (r.decode_tokens > 0) {
            const double tps = static_cast<double>(r.decode_tokens) / r.forward_seconds;
            std::printf("decode_tps     = %.2f (decode_tokens / total_forward)\n", tps);
        }
        std::printf("seeds          = [");
        for (size_t i = 0; i < seeds.size(); ++i) {
            std::printf("%s%d", i ? ", " : "", seeds[i]);
        }
        std::printf("]\n");
        std::printf("generated      = [");
        for (size_t i = 0; i < r.generated_tokens.size(); ++i) {
            std::printf("%s%d", i ? ", " : "", r.generated_tokens[i]);
        }
        std::printf("]\n");
        std::printf("top_logits[first..last] = ");
        for (size_t i = 0; i < r.top_logits.size(); ++i) {
            std::printf("%s%.3f", i ? ", " : "", r.top_logits[i]);
        }
        std::printf("\n");

        if (static_cast<int>(r.generated_tokens.size()) != max_new) {
            std::cerr << "[FAIL] generated_tokens size mismatch\n"; return 1;
        }
        for (int t : r.generated_tokens) {
            if (t < 0 || t >= r.vocab) {
                std::cerr << "[FAIL] generated token out of vocab\n"; return 1;
            }
        }
        std::cout << "[PASS] gguf greedy decode smoke\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
