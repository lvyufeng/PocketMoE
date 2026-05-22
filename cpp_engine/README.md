# dsv4 C++ Engine

Standalone C++/CUDA inference-engine experiment for DeepSeek V4 Flash.

This directory is intentionally separate from the existing PyTorch runtime and torch extensions. The first milestone only builds a CLI, parses GGUF files, dumps model metadata/config, and hosts raw-pointer CUDA substrate tests.

## Build

```bash
cmake -S cpp_engine -B build/cpp_engine -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp_engine -j
```

## Inspect GGUF

```bash
build/cpp_engine/tools/inspect_gguf /path/to/model.gguf --limit 20
```

## CLI skeleton

```bash
build/cpp_engine/dsv4_cpp_engine --model /path/to/model.gguf --dump-config
```
