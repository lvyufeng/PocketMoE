#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace dsv4 {

class Tokenizer {
public:
    explicit Tokenizer(const std::string& ckpt_dir);

    std::string token(int id) const;
    std::string decode_piece(int id) const;
    std::string decode_tokens(const std::vector<int>& ids) const;
    std::vector<int> encode_basic(const std::string& text, bool add_bos = true) const;
    size_t vocab_size() const { return id_to_token_.size(); }

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::map<std::pair<std::string, std::string>, int> merge_rank_;
};

}  // namespace dsv4
