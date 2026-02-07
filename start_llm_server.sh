#!/bin/bash
# start_llm_server.sh

echo "🚀 Starting MedGemma-TCCC Server..."

./llama.cpp/build/bin/llama-server \
  -m ./models/medgemma-tccc-q4.gguf \
  -ngl 99 \
  -c 2048 \
  -b 512 \
  --no-mmap \
  -t 4 \
  --port 8080 \
  --host 127.0.0.1

# Keep running until Ctrl+C
# chmod +x start_llm_server.sh
# ./start_llm_server.sh
