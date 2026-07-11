#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${1:-$ROOT/models}"
MODEL_NAME="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
BASE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

if [[ ! -f "$MODEL_NAME/model.int8.onnx" || ! -f "$MODEL_NAME/tokens.txt" ]]; then
    # Resume partial downloads; model archives are large and field networks can be slow.
    curl -fL -C - --retry 10 --retry-all-errors \
        -o "$MODEL_NAME.tar.bz2" "$BASE_URL/$MODEL_NAME.tar.bz2"
    bzip2 -t "$MODEL_NAME.tar.bz2"
    tar xjf "$MODEL_NAME.tar.bz2"
    rm "$MODEL_NAME.tar.bz2"
fi

if [[ ! -f silero_vad.onnx ]]; then
    curl -fL -C - --retry 10 --retry-all-errors \
        -o silero_vad.onnx "$BASE_URL/silero_vad.onnx"
fi

echo "模型已就绪: $MODEL_DIR"
