#!/bin/bash
# start_llm_server.sh

echo "🚀 Starting MedGemma-TCCC Server..."

/Users/fiercecoyote/llama.cpp/build/bin/llama-server \
  -m /Users/fiercecoyote/medevac-gemma/models/medgemma/medgemma-1.5-4b-tccc-lora-q4.gguf \
  -ngl 99 \
  -c 1024 \
  -b 512 \
  -ub 256 \
  --no-mmap \
  -t 6 \
  -tb 6 \
  --cont-batching \
  -fa on\
  --port 8080 \
  --host 127.0.0.1

# Keep running until Ctrl+C
# chmod +x start_llm_server.sh
# ./start_llm_server.sh
