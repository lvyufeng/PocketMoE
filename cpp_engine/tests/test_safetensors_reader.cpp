#include "safetensors_reader.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

void require_tensor(const dsv4::SafeTensorsShard& shard, const std::string& name, dsv4::SafeDType dtype, std::initializer_list<uint64_t> shape) {
    const auto* info = shard.find_tensor(name);
    require(info != nullptr, "missing tensor: " + name);
    require(info->dtype == dtype, "bad dtype for " + name);
    require(info->shape == std::vector<uint64_t>(shape), "bad shape for " + name);
    require(shard.tensor_data(*info) != nullptr, "null tensor data: " + name);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_safetensors_reader <ckpt_dir>\n";
        return 2;
    }
    try {
        std::string ckpt = argv[1];
        dsv4::SafeTensorsIndex index(ckpt);
        require(index.tensor_count() == 69187, "unexpected tensor count");
        require(index.shard_count() == 46, "unexpected shard count");
        require(index.shard_for_tensor("embed.weight") != nullptr, "embed.weight missing from index");

        dsv4::SafeTensorsShard shard1(index.shard_path("model-00001-of-00046.safetensors"));
        require_tensor(shard1, "embed.weight", dsv4::SafeDType::BF16, {129280, 4096});

        dsv4::SafeTensorsShard shard2(index.shard_path("model-00002-of-00046.safetensors"));
        require_tensor(shard2, "layers.0.ffn.gate.tid2eid", dsv4::SafeDType::I64, {129280, 6});
        require_tensor(shard2, "layers.0.attn.wkv.weight", dsv4::SafeDType::F8_E4M3, {512, 4096});
        require_tensor(shard2, "layers.0.attn.wkv.scale", dsv4::SafeDType::F8_E8M0, {4, 32});

        std::cout << "[PASS] safetensors_reader tensors=" << index.tensor_count() << " shards=" << index.shard_count() << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
