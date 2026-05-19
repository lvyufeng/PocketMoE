#include "tokenizer.hpp"

#include "json_lite.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace dsv4 {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

uint64_t json_u64(const JsonValue& value) {
    const double n = value.number();
    if (n < 0) throw std::runtime_error("negative tokenizer id");
    return static_cast<uint64_t>(n + 0.5);
}

void set_token(std::vector<std::string>& id_to_token, uint64_t id, const std::string& token) {
    if (id >= id_to_token.size()) id_to_token.resize(static_cast<size_t>(id) + 1);
    id_to_token[static_cast<size_t>(id)] = token;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

std::string utf8_codepoint(uint32_t cp) {
    std::string out;
    if (cp <= 0x7f) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
    return out;
}

std::vector<std::string> byte_alphabet() {
    std::vector<uint32_t> bs;
    for (uint32_t c = '!'; c <= '~'; ++c) bs.push_back(c);
    for (uint32_t c = 0xA1; c <= 0xAC; ++c) bs.push_back(c);
    for (uint32_t c = 0xAE; c <= 0xFF; ++c) bs.push_back(c);
    std::vector<uint32_t> cs = bs;
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }
    std::vector<std::string> out(256);
    for (size_t i = 0; i < bs.size(); ++i) out[bs[i]] = utf8_codepoint(cs[i]);
    return out;
}

std::pair<std::string, std::string> split_merge(const std::string& merge) {
    const size_t pos = merge.find(' ');
    if (pos == std::string::npos) throw std::runtime_error("bad BPE merge: " + merge);
    return {merge.substr(0, pos), merge.substr(pos + 1)};
}

std::unordered_map<std::string, unsigned char> byte_decoder() {
    std::unordered_map<std::string, unsigned char> out;
    const auto alphabet = byte_alphabet();
    for (size_t i = 0; i < alphabet.size(); ++i) out[alphabet[i]] = static_cast<unsigned char>(i);
    return out;
}

bool starts_with_at(const std::string& s, size_t pos, const std::string& prefix) {
    return pos + prefix.size() <= s.size() && s.compare(pos, prefix.size(), prefix) == 0;
}

}  // namespace

Tokenizer::Tokenizer(const std::string& ckpt_dir) {
    JsonValue root_value = parse_json(read_file(ckpt_dir + "/tokenizer.json"));
    const JsonObject& root = root_value.object();
    const JsonObject& model = object_get(root, "model")->object();
    const JsonObject& vocab = object_get(model, "vocab")->object();
    for (const auto& [tok, id_value] : vocab) {
        const uint64_t id = json_u64(id_value);
        set_token(id_to_token_, id, tok);
        token_to_id_[tok] = static_cast<int>(id);
    }
    if (const JsonValue* merges = object_get(model, "merges")) {
        int rank = 0;
        for (const JsonValue& item : merges->array()) {
            merge_rank_[split_merge(item.string())] = rank++;
        }
    }
    if (const JsonValue* added = object_get(root, "added_tokens")) {
        for (const JsonValue& item : added->array()) {
            const JsonObject& obj = item.object();
            const JsonValue* id = object_get(obj, "id");
            const JsonValue* content = object_get(obj, "content");
            if (id != nullptr && content != nullptr) {
                const uint64_t token_id = json_u64(*id);
                set_token(id_to_token_, token_id, content->string());
                token_to_id_[content->string()] = static_cast<int>(token_id);
            }
        }
    }
}

std::string Tokenizer::token(int id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return "<invalid>";
    const std::string& tok = id_to_token_[static_cast<size_t>(id)];
    return tok.empty() ? "<empty>" : tok;
}

std::string Tokenizer::decode_piece(int id) const {
    std::string s = token(id);
    s = replace_all(s, "Ġ", " ");
    s = replace_all(s, "▁", " ");
    s = replace_all(s, "Ċ", "\n");
    return s;
}

std::string Tokenizer::decode_tokens(const std::vector<int>& ids) const {
    static const auto decoder = byte_decoder();
    std::string out;
    for (int id : ids) {
        std::string piece = token(id);
        if (piece.size() >= 2 && piece.front() == '<' && piece.back() == '>') continue;
        for (size_t i = 0; i < piece.size();) {
            bool matched = false;
            for (const auto& [encoded, byte] : decoder) {
                if (starts_with_at(piece, i, encoded)) {
                    out.push_back(static_cast<char>(byte));
                    i += encoded.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) out.push_back(piece[i++]);
        }
    }
    return out;
}

std::vector<int> Tokenizer::encode_basic(const std::string& text, bool add_bos) const {
    static const std::vector<std::string> bytes = byte_alphabet();
    std::vector<int> ids;
    if (add_bos) ids.push_back(0);
    std::vector<std::string> pieces;
    pieces.reserve(text.size());
    for (unsigned char c : text) pieces.push_back(bytes[c]);
    while (pieces.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_pos = pieces.size();
        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            auto it = merge_rank_.find({pieces[i], pieces[i + 1]});
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }
        if (best_pos == pieces.size()) break;
        pieces[best_pos] += pieces[best_pos + 1];
        pieces.erase(pieces.begin() + static_cast<long>(best_pos + 1));
    }
    for (const auto& piece : pieces) {
        auto it = token_to_id_.find(piece);
        if (it == token_to_id_.end()) throw std::runtime_error("BPE piece missing from vocab: " + piece);
        ids.push_back(it->second);
    }
    return ids;
}

}  // namespace dsv4
