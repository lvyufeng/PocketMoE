#include "tokenizer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

void check_encode(const dsv4::Tokenizer& tok, const std::string& text, const std::vector<int>& expected) {
    auto ids = tok.encode_basic(text, false);
    if (ids != expected) {
        std::string got;
        for (int id : ids) got += std::to_string(id) + " ";
        std::string want;
        for (int id : expected) want += std::to_string(id) + " ";
        throw std::runtime_error("encode mismatch for [" + text + "] got=[" + got + "] want=[" + want + "]");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_tokenizer <ckpt_dir>\n";
        return 2;
    }
    try {
        dsv4::Tokenizer tok(argv[1]);
        require(tok.vocab_size() >= 129280, "bad vocab size");
        require(tok.token(0) == "<｜begin▁of▁sentence｜>", "bad bos token");
        require(tok.token(1) == "<｜end▁of▁sentence｜>", "bad eos token");
        require(!tok.token(107590).empty(), "missing generated token");
        check_encode(tok, "hello", {33310});
        check_encode(tok, " hello", {44388});
        check_encode(tok, "two", {23315});
        check_encode(tok, " two", {1234});
        check_encode(tok, "Hello world", {19923, 2058});
        check_encode(tok, "123", {6895});
        auto bos = tok.encode_basic("hello", true);
        require(bos.size() == 2 && bos[0] == 0 && bos[1] == 33310, "bad add_bos encode");
        require(tok.decode_tokens(tok.encode_basic("Hello world", true)) == "Hello world", "bad decode roundtrip");
        require(tok.decode_tokens({55262}) == "我们必须", "bad byte-level decode");
        std::cout << "[PASS] tokenizer vocab=" << tok.vocab_size()
                  << " token107590=" << tok.decode_piece(107590) << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
