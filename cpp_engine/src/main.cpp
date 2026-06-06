#include "cuda_ops.hpp"
#include "dsv4_engine.hpp"
#include "model_config.hpp"
#include "openai_server.hpp"
#include "persistent_engine.hpp"
#include "python_sidecar.hpp"
#include "safetensors_reader.hpp"
#include "tokenizer.hpp"
#include "tp_comm.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

namespace {

struct Args {
    std::string model;
    std::string ckpt;
    std::string inspect_tensor;
    std::string prompt;
    std::string token_ids_csv;
    std::string token_ids_file;
    int smoke_layers = 1;
    int forward_token = -1;
    int position = 0;
    int max_new_tokens = 1;
    int tp_world = 1;
    int tp_rank = 0;
    int device = -1;
    std::string nccl_id_path;
    bool generate_token = false;
    bool resident_bench = false;
    bool dump_config = false;
    bool inspect = false;
    bool smoke_forward = false;
    bool use_persistent = false;
    int max_context = 0;
    bool serve = false;
    int port = 8000;
    std::string host = "0.0.0.0";
    std::string python_bin = "python";
    std::string sidecar_script;
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
        } else if (arg == "--resident-bench") {
            args.smoke_forward = true;
            args.generate_token = true;
            args.resident_bench = true;
        } else if (arg == "--prompt" && i + 1 < argc) {
            args.smoke_forward = true;
            args.prompt = argv[++i];
        } else if (arg == "--token-ids" && i + 1 < argc) {
            args.smoke_forward = true;
            args.token_ids_csv = argv[++i];
        } else if (arg == "--token-ids-file" && i + 1 < argc) {
            args.smoke_forward = true;
            args.token_ids_file = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            ++i;
        } else if (arg == "--max-new-tokens" && i + 1 < argc) {
            args.max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--tp-world" && i + 1 < argc) {
            args.tp_world = std::stoi(argv[++i]);
        } else if (arg == "--tp-rank" && i + 1 < argc) {
            args.tp_rank = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            args.device = std::stoi(argv[++i]);
        } else if (arg == "--nccl-id-path" && i + 1 < argc) {
            args.nccl_id_path = argv[++i];
        } else if (arg == "--use-persistent") {
            args.use_persistent = true;
        } else if (arg == "--max-context" && i + 1 < argc) {
            args.max_context = std::stoi(argv[++i]);
        } else if (arg == "--serve") {
            args.serve = true;
        } else if (arg == "--port" && i + 1 < argc) {
            args.port = std::stoi(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            args.host = argv[++i];
        } else if (arg == "--python" && i + 1 < argc) {
            args.python_bin = argv[++i];
        } else if (arg == "--sidecar" && i + 1 < argc) {
            args.sidecar_script = argv[++i];
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (args.ckpt.empty() && !args.model.empty() && is_dir(args.model) && path_exists(args.model + "/model.safetensors.index.json")) {
        args.ckpt = args.model;
        args.model.clear();
    }
    if (args.tp_world <= 0) throw std::runtime_error("--tp-world must be positive");
    if (args.tp_rank < 0 || args.tp_rank >= args.tp_world) throw std::runtime_error("--tp-rank must be in [0, tp_world)");
    if (args.model.empty() && args.ckpt.empty()) {
        throw std::runtime_error("--model or --ckpt is required");
    }
    return args;
}


std::vector<int> parse_token_ids_csv(const std::string& text) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        out.push_back(std::stoi(item));
    }
    return out;
}

std::vector<int> load_token_ids(const Args& args) {
    if (!args.token_ids_csv.empty()) return parse_token_ids_csv(args.token_ids_csv);
    if (!args.token_ids_file.empty()) {
        std::ifstream in(args.token_ids_file);
        if (!in) throw std::runtime_error("failed to open token ids file: " + args.token_ids_file);
        std::stringstream buf;
        buf << in.rdbuf();
        return parse_token_ids_csv(buf.str());
    }
    return {};
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
        if (args.device >= 0) {
            if (cudaSetDevice(args.device) != cudaSuccess) throw std::runtime_error("failed to set CUDA device");
        } else if (args.tp_world > 1) {
            if (cudaSetDevice(args.tp_rank) != cudaSuccess) throw std::runtime_error("failed to set CUDA device for tp rank");
        }
        if (args.tp_world > 1) {
            std::cout << "tp_world=" << args.tp_world << " tp_rank=" << args.tp_rank
                      << " device=" << (args.device >= 0 ? args.device : args.tp_rank) << "\n";
        }
        if (args.serve) {
            if (args.ckpt.empty()) throw std::runtime_error("--serve requires --ckpt");
            const int layer_count = args.smoke_layers > 0 ? args.smoke_layers : 43;
            const int max_context = args.max_context > 0 ? args.max_context : 8192;
            dsv4::ForwardSmokeOptions opts;
            opts.tp_world = args.tp_world;
            opts.tp_rank = args.tp_rank;
            opts.device = args.device >= 0 ? args.device : args.tp_rank;
            opts.nccl_id_path = args.nccl_id_path;
            dsv4::PersistentEngine engine(args.ckpt, opts, layer_count, max_context);
            engine.warmup_tp();
            if (args.tp_rank > 0) {
                // Worker rank: park on the NCCL command channel until rank 0
                // sends SHUTDOWN.
                engine.run_worker_loop();
                return 0;
            }
            const std::string sidecar_script = args.sidecar_script.empty()
                ? std::string("src/server/cpp_sidecar.py")
                : args.sidecar_script;
            dsv4::PythonSidecar sidecar(args.python_bin, sidecar_script, args.ckpt);
            dsv4::OpenAIServerConfig cfg;
            cfg.port = args.port;
            cfg.host = args.host;
            dsv4::OpenAIServer server(engine, sidecar, cfg);
            server.run();
            engine.worker_command_shutdown();
            return 0;
        }
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
                std::vector<int> prompt_ids = load_token_ids(args);
                if (!prompt_ids.empty()) {
                    args.forward_token = prompt_ids.back();
                    args.position = static_cast<int>(prompt_ids.size()) - 1;
                    std::cout << "prompt_tokens=" << prompt_ids.size()
                              << " last_token=" << args.forward_token
                              << " position=" << args.position
                              << " token_ids_source=" << (args.token_ids_csv.empty() ? "file" : "csv") << "\n";
                }
                if (!args.prompt.empty()) {
                    prompt_ids = tokenizer.encode_basic(args.prompt, false);
                    if (prompt_ids.empty()) throw std::runtime_error("prompt encoded to no tokens");
                    args.forward_token = prompt_ids.back();
                    args.position = static_cast<int>(prompt_ids.size()) - 1;
                    std::cout << "prompt_tokens=" << prompt_ids.size()
                              << " last_token=" << args.forward_token
                              << " position=" << args.position
                              << " last_text=" << tokenizer.decode_piece(args.forward_token) << "\n";
                    if (!args.generate_token) {
                        dsv4::ForwardSmokeOptions opts;
                        opts.tp_world = args.tp_world;
                        opts.tp_rank = args.tp_rank;
                        opts.device = args.device >= 0 ? args.device : args.tp_rank;
                        opts.nccl_id_path = args.nccl_id_path;
                        dsv4::ForwardSmokeResult result = dsv4::run_safetensors_prompt_forward_with_options(args.ckpt, prompt_ids, args.smoke_layers, opts);
                        int top_token = result.top_token;
                        float top_logit = result.top_logit;
#ifdef DSV4_HAVE_NCCL
                        if (args.tp_world > 1 && !args.nccl_id_path.empty()) {
                            dsv4::TpTopResult global = dsv4::nccl_global_top1(
                                args.tp_world,
                                args.tp_rank,
                                args.device >= 0 ? args.device : args.tp_rank,
                                args.nccl_id_path.c_str(),
                                result.top_token,
                                result.top_logit);
                            top_token = global.token;
                            top_logit = global.logit;
                        }
#endif
                        std::cout << "smoke_forward=1 token=" << result.token
                                  << " layers=" << result.layers
                                  << " dim=" << result.dim
                                  << " inter=" << result.inter
                                  << " logits=" << result.logits
                                  << " tp_world=" << args.tp_world
                                  << " tp_rank=" << args.tp_rank
                                  << " local_top_token=" << result.top_token
                                  << " local_top_logit=" << result.top_logit
                                  << " top_token=" << top_token
                                  << " top_logit=" << top_logit
                                  << " checksum=" << result.checksum << "\n";
                        return 0;
                    }
                }
                if (args.generate_token) {
                    if (prompt_ids.empty()) {
                        if (args.forward_token < 0) throw std::runtime_error("--generate-token or --prompt is required for generation");
                        prompt_ids.push_back(args.forward_token);
                    }
                    dsv4::ForwardSmokeOptions opts;
                    opts.tp_world = args.tp_world;
                    opts.tp_rank = args.tp_rank;
                    opts.device = args.device >= 0 ? args.device : args.tp_rank;
                    opts.nccl_id_path = args.nccl_id_path;
                    if (args.use_persistent) {
                        const int max_context = args.max_context > 0
                            ? args.max_context
                            : static_cast<int>(prompt_ids.size()) + args.max_new_tokens;
                        dsv4::PersistentEngine engine(args.ckpt, opts, args.smoke_layers, max_context);
                        engine.warmup_tp();
                        if (args.tp_rank > 0) {
                            engine.run_worker_loop();
                            return 0;
                        }
                        dsv4::SamplingParams sp;
                        sp.greedy = true;
                        std::vector<int> generated_ids;
                        generated_ids.reserve(static_cast<size_t>(args.max_new_tokens));
                        using Clock = std::chrono::steady_clock;
                        const auto t_total0 = Clock::now();
                        engine.worker_command_reset();
                        engine.reset_session();
                        const auto t_prefill0 = Clock::now();
                        engine.worker_command_prefill(prompt_ids);
                        int token = engine.prefill(prompt_ids, sp);
                        const auto t_prefill1 = Clock::now();
                        generated_ids.push_back(token);
                        int position = static_cast<int>(prompt_ids.size());
                        const auto t_decode0 = Clock::now();
                        for (int step = 1; step < args.max_new_tokens; ++step) {
                            engine.worker_command_decode(token, position + step - 1);
                            token = engine.decode_step(token, position + step - 1, sp);
                            generated_ids.push_back(token);
                        }
                        const auto t_decode1 = Clock::now();
                        engine.worker_command_shutdown();
                        auto sec = [](auto a, auto b) {
                            return std::chrono::duration<double>(b - a).count();
                        };
                        for (size_t step = 0; step < generated_ids.size(); ++step) {
                            std::cout << "generate_step=" << step
                                      << " token=" << generated_ids[step]
                                      << " token_text=" << tokenizer.decode_tokens({generated_ids[step]})
                                      << " decoded=" << tokenizer.decode_tokens(std::vector<int>(generated_ids.begin(), generated_ids.begin() + step + 1)) << "\n";
                        }
                        if (args.resident_bench) {
                            const double wall = sec(t_total0, t_decode1);
                            const double prefill = sec(t_prefill0, t_prefill1);
                            const double decode = sec(t_decode0, t_decode1);
                            const double prefill_tps = prefill > 0.0 ? static_cast<double>(prompt_ids.size()) / prefill : 0.0;
                            const int decoded = args.max_new_tokens > 1 ? args.max_new_tokens - 1 : 0;
                            const double decode_tps = decode > 0.0 ? static_cast<double>(decoded) / decode : 0.0;
                            const double total_tokens = static_cast<double>(prompt_ids.size() + generated_ids.size());
                            const double tps = wall > 0.0 ? total_tokens / wall : 0.0;
                            std::cout << "resident_bench=1 prompt_tokens=" << prompt_ids.size()
                                      << " generated_tokens=" << generated_ids.size()
                                      << " wall=" << wall
                                      << " tokens_per_s=" << tps
                                      << " prefill_seconds=" << prefill
                                      << " prefill_tokens_per_s=" << prefill_tps
                                      << " decode_seconds=" << decode
                                      << " decode_tokens_per_s=" << decode_tps
                                      << " decode_token_count=" << decoded
                                      << " tp_world=" << args.tp_world
                                      << " tp_rank=" << args.tp_rank
                                      << " use_persistent=1\n";
                        }
                        return 0;
                    }
                    auto timed = dsv4::run_safetensors_generate_tokens_timed_with_options(args.ckpt, prompt_ids, args.smoke_layers, args.max_new_tokens, opts);
                    auto& results = timed.tokens;
                    std::vector<int> generated_ids;
                    generated_ids.reserve(results.size());
                    for (size_t step = 0; step < results.size(); ++step) {
                        const dsv4::ForwardSmokeResult& result = results[step];
                        generated_ids.push_back(result.token);
                        std::cout << "generate_step=" << step
                                  << " token=" << result.token
                                  << " token_text=" << tokenizer.decode_tokens({result.token})
                                  << " top_token=" << result.top_token
                                  << " top_text=" << tokenizer.decode_tokens({result.top_token})
                                  << " decoded=" << tokenizer.decode_tokens(generated_ids)
                                  << " top_logit=" << result.top_logit
                                  << " checksum=" << result.checksum << "\n";
                    }
                    if (args.resident_bench) {
                        const double wall = timed.wall_seconds;
                        const double prefill = timed.prefill_seconds;
                        const double decode = timed.decode_seconds;
                        const double prefill_tps = prefill > 0.0 ? static_cast<double>(timed.prompt_tokens) / prefill : 0.0;
                        const double decode_tps = decode > 0.0 ? static_cast<double>(timed.decode_tokens) / decode : 0.0;
                        const double tokens = static_cast<double>(prompt_ids.size() + results.size());
                        const double tps = wall > 0.0 ? tokens / wall : 0.0;
                        std::cout << "resident_bench=1 prompt_tokens=" << prompt_ids.size()
                                  << " generated_tokens=" << results.size()
                                  << " wall=" << wall
                                  << " tokens_per_s=" << tps
                                  << " prefill_seconds=" << prefill
                                  << " prefill_tokens_per_s=" << prefill_tps
                                  << " decode_seconds=" << decode
                                  << " decode_tokens_per_s=" << decode_tps
                                  << " decode_token_count=" << timed.decode_tokens
                                  << " tp_world=" << args.tp_world
                                  << " tp_rank=" << args.tp_rank << "\n";
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
