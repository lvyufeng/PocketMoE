#include "cuda_ops.hpp"
#include "dsv4_engine.hpp"
#include "model_config.hpp"
#include "safetensors_reader.hpp"
#include "tokenizer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

namespace {

struct Args {
    std::string model;
    std::string ckpt;
    std::string inspect_tensor;
    std::string prompt;
    int smoke_layers = 1;
    int forward_token = -1;
    int position = 0;
    int max_new_tokens = 1;
    bool generate_token = false;
    bool dump_config = false;
    bool inspect = false;
    bool smoke_forward = false;
};

bool path_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

bool is_dir(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            args.model = argv[++i];
        } else if (arg == "--ckpt" && i + 1 < argc) {
            args.ckpt = argv[++i];
        } else if (arg == "--inspect-tensor" && i + 1 < argc) {
            args.inspect_tensor = argv[++i];
        } else if (arg == "--dump-config") {
            args.dump_config = true;
        } else if (arg == "--inspect") {
            args.inspect = true;
        } else if (arg == "--smoke-forward") {
            args.smoke_forward = true;
        } else if (arg == "--smoke-layers" && i + 1 < argc) {
            args.smoke_forward = true;
            args.smoke_layers = std::stoi(argv[++i]);
        } else if (arg == "--forward-token" && i + 1 < argc) {
            args.smoke_forward = true;
            args.forward_token = std::stoi(argv[++i]);
        } else if (arg == "--position" && i + 1 < argc) {
            args.position = std::stoi(argv[++i]);
        } else if (arg == "--generate-token" && i + 1 < argc) {
            args.smoke_forward = true;
            args.generate_token = true;
            args.forward_token = std::stoi(argv[++i]);
        } else if (arg == "--prompt" && i + 1 < argc) {
            args.smoke_forward = true;
            args.prompt = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            ++i;
        } else if (arg == "--max-new-tokens" && i + 1 < argc) {
            args.max_new_tokens = std::stoi(argv[++i]);
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (args.ckpt.empty() && !args.model.empty() && is_dir(args.model) && path_exists(args.model + "/model.safetensors.index.json")) {
        args.ckpt = args.model;
        args.model.clear();
    }
    if (args.model.empty() && args.ckpt.empty()) {
        throw std::runtime_error("--model or --ckpt is required");
    }
    return args;
}

void print_safe_tensor(const dsv4::SafeTensorInfo& info, const std::string& shard) {
    std::cout << info.name << " dtype=" << dsv4::safe_dtype_name(info.dtype) << " shape=[";
    for (size_t i = 0; i < info.shape.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << info.shape[i];
    }
    std::cout << "] begin=" << info.data_begin << " abs=" << info.absolute_begin << " bytes=" << info.nbytes << " shard=" << shard << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        if (!args.ckpt.empty()) {
            dsv4::SafeTensorsIndex index(args.ckpt);
            std::cout << "dsv4_cpp_engine opened " << args.ckpt << "\n";
            std::cout << "format=safetensors tensors=" << index.tensor_count()
                      << " shards=" << index.shard_count()
                      << " total_size=" << index.total_size()
                      << " cuda=" << (dsv4::cuda_runtime_available() ? "yes" : "no") << "\n";
            if (args.dump_config) {
                std::cout << dsv4::ModelConfig::from_hf_config(args.ckpt).to_string();
            }
            if (!args.inspect_tensor.empty()) {
                const std::string* shard_name = index.shard_for_tensor(args.inspect_tensor);
                if (shard_name == nullptr) throw std::runtime_error("tensor not found: " + args.inspect_tensor);
                dsv4::SafeTensorsShard shard(index.shard_path(*shard_name));
                const auto* info = shard.find_tensor(args.inspect_tensor);
                if (info == nullptr) throw std::runtime_error("tensor missing in shard header: " + args.inspect_tensor);
                print_safe_tensor(*info, *shard_name);
            }
            if (args.smoke_forward) {
                dsv4::Tokenizer tokenizer(args.ckpt);
                if (!args.prompt.empty()) {
                    auto ids = tokenizer.encode_basic(args.prompt, true);
                    if (ids.empty()) throw std::runtime_error("prompt encoded to no tokens");
                    args.forward_token = ids.back();
                    args.position = static_cast<int>(ids.size()) - 1;
                    std::cout << "prompt_tokens=" << ids.size()
                              << " last_token=" << args.forward_token
                              << " position=" << args.position
                              << " last_text=" << tokenizer.decode_piece(args.forward_token) << "\n";
                    if (!args.generate_token) {
                        dsv4::ForwardSmokeResult result = dsv4::run_safetensors_prompt_forward(args.ckpt, ids, args.smoke_layers);
                        std::cout << "smoke_forward=1 token=" << result.token
                                  << " layers=" << result.layers
                                  << " dim=" << result.dim
                                  << " inter=" << result.inter
                                  << " logits=" << result.logits
                                  << " top_token=" << result.top_token
                                  << " top_logit=" << result.top_logit
                                  << " checksum=" << result.checksum << "\n";
                        return 0;
                    }
                }
                if (args.generate_token) {
                    int token = args.forward_token;
                    if (token < 0) throw std::runtime_error("--generate-token or --prompt is required for generation");
                    for (int step = 0; step < args.max_new_tokens; ++step) {
                        dsv4::ForwardSmokeResult result = dsv4::run_safetensors_token_forward_at_position(args.ckpt, token, args.smoke_layers, args.position + step);
                        std::cout << "generate_step=" << step
                                  << " token=" << result.token
                                  << " token_text=" << tokenizer.decode_piece(result.token)
                                  << " top_token=" << result.top_token
                                  << " top_text=" << tokenizer.decode_piece(result.top_token)
                                  << " top_logit=" << result.top_logit
                                  << " checksum=" << result.checksum << "\n";
                        token = result.top_token;
                    }
                } else {
                    dsv4::ForwardSmokeResult result = args.forward_token >= 0
                        ? dsv4::run_safetensors_token_forward(args.ckpt, args.forward_token, args.smoke_layers)
                        : dsv4::run_safetensors_layer_loop_smoke(args.ckpt, args.smoke_layers);
                    std::cout << "smoke_forward=1 token=" << result.token
                              << " layers=" << result.layers
                              << " dim=" << result.dim
                              << " inter=" << result.inter
                              << " logits=" << result.logits
                              << " top_token=" << result.top_token
                              << " top_logit=" << result.top_logit
                              << " checksum=" << result.checksum << "\n";
                }
            }
            if (!args.dump_config && args.inspect_tensor.empty() && !args.inspect && !args.smoke_forward) {
                std::cout << "inference_not_implemented=1\n";
            }
            return 0;
        }
        dsv4::Dsv4Engine engine(args.model);
        std::cout << "dsv4_cpp_engine opened " << args.model << "\n";
        std::cout << "format=gguf gguf_version=" << engine.gguf().version()
                  << " tensors=" << engine.gguf().tensor_count()
                  << " metadata=" << engine.gguf().metadata_count()
                  << " alignment=" << engine.gguf().alignment()
                  << " cuda=" << (dsv4::cuda_runtime_available() ? "yes" : "no") << "\n";
        if (args.dump_config) {
            std::cout << engine.config().to_string();
        }
        if (!args.dump_config && !args.inspect) {
            std::cout << "inference_not_implemented=1\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
