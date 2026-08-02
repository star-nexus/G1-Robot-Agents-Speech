#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${1:-$ROOT/models}"
MODEL_NAME="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
BASE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

if [[ ! -f "$MODEL_NAME/model.onnx" || ! -f "$MODEL_NAME/tokens.txt" ]]; then
    curl -fL -C - --retry 10 --retry-all-errors \
        -o "$MODEL_NAME.tar.bz2" "$BASE_URL/$MODEL_NAME.tar.bz2"
    bzip2 -t "$MODEL_NAME.tar.bz2"
    tar xjf "$MODEL_NAME.tar.bz2"
    rm "$MODEL_NAME.tar.bz2"
fi

echo "GPU FP32 模型已就绪: $MODEL_DIR/$MODEL_NAME/model.onnx"
