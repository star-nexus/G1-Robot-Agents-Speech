# Qwen3-TTS with vLLM-Omni on JetPack 6.2

## Status

The speech service now contains a transport-neutral streaming TTS client, sentence
scheduler, ALSA player, render-reference tee, and hard barge-in path. The server
runtime remains isolated from the stable ASR environments.

The JetPack 6.2 server experiment is **not yet a runnable production baseline**.
The native build, import, CLI, SM87 FlashAttention 2 selection, Qwen talker weight
load, and vLLM graph compilation all succeed. Startup then stops at a second
vLLM-Omni/vLLM internal API mismatch before the HTTP/WebSocket port becomes ready.
No TTS latency or audio-quality result is claimed from this experiment.

The JetPack 6.2 source-build experiment uses:

| Component | Selection |
|---|---|
| Platform | Jetson Orin NX 16 GB, SM 8.7, CUDA 12.6 |
| vLLM-Omni | upstream `main`, commit `72a02b492a6957acc4659fde09be85c00ad0f90b` |
| vLLM | source commit `568afb3a1`, reported as `0.26.1.dev0`, source build for SM 8.7 |
| Python environment | separate host-local virtual environment |
| PyTorch | Jetson ARM64 2.11.0, CUDA 12.6 |
| Triton | 3.5.1 |
| torchaudio | matching 2.11.0 source build, isolated environment only |
| Model | local Qwen3-TTS-12Hz-0.6B-CustomVoice |

This combination is substantially newer than the project's sealed ASR PyTorch 2.5
environment. It must not be installed into `.venv-gpu`. Latest vLLM's ordinary
server wheels target a different CUDA/platform matrix; this experiment therefore
builds the CUDA extensions locally with `CUDA_HOME=/usr/local/cuda` and
`TORCH_CUDA_ARCH_LIST=8.7`.

## Reproducing the isolated build

Use a separate root and a Jetson-native PyTorch wheel. The important source-build
settings are:

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.7
export VLLM_TARGET_DEVICE=cuda
export VLLM_ORIN_TTS_MINIMAL=1
export MAX_JOBS=2
export NVCC_THREADS=1
export CMAKE_BUILD_TYPE=Release
export CMAKE_ARGS="-DVLLM_ORIN_TTS_MINIMAL=ON"
```

vLLM 0.26 requires a newer CMake than Ubuntu 22.04's 3.22; install it into the new
virtual environment, not the OS. Build dependency resolution must reuse the already
selected Jetson PyTorch rather than downloading generic SBSA/server PyTorch. The
Jetson PyTorch 2.11 package also needs its matching cuDSS runtime on
`LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/path/to/libcudss/12${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

An environment created with JetPack system packages visible can also see the
old system `torchvision`, whose compiled operators do not match PyTorch 2.11.
Qwen3-TTS is text/audio-only, so the provided launcher marks torchvision as an
unavailable optional Transformers feature before importing vLLM-Omni. Set
`VLLM_OMNI_DISABLE_TORCHVISION=0` only after installing a Jetson-native
torchvision that exactly matches the isolated PyTorch.

The source checkout configured successfully for CUDA 12.6 and native SM 8.7. Some
optional CUDA 12.8-only kernels (DeepGEMM, FlashMLA, Qutlass) are correctly omitted.
The upstream build graph requested 357 compilation units on this host, dominated by
Hopper FA3, generic FA2 variants, and an independent MoE extension that dense
Qwen3-TTS does not import. Apply `patches/vllm-0.26-orin-tts-minimal.patch` and
enable both the environment variable and CMake option above to skip FA3 and that
MoE target without deleting symbols from vLLM's common binding module. Apply
`patches/vllm-flash-attn-sm87-qwen3-tts.patch` to vLLM's pinned
`vllm-project/flash-attention` checkout and pass that checkout through
`VLLM_FLASH_ATTN_SRC_DIR`; it disables Hopper-only FA3 and retains FP16/BF16 FA2
dense and split-KV forward instantiations for Qwen3-TTS's `head_dim=128`, while
excluding backward, sparse, and unrelated head dimensions. Both ordinary and
split entry points are required by the loaded FA2 shared object. This is a target-specific source
patch, so the resulting wheel must not be presented as a general-purpose vLLM
wheel. The final clean build graph is roughly 64 native steps instead of 357 and
preserves the full common binding implementation. Two more aggressive reductions
were rejected during import testing: one removed symbols from vLLM's common ABI,
and another left unresolved FA2 head-dimension templates. No ASR environment is
modified if this experiment fails.

Latest vLLM-Omni main also contains one stale call to vLLM's input-randomization
context manager. Apply `patches/vllm-omni-vllm026-randomize-inputs.patch` to the
vLLM-Omni checkout. It removes the obsolete keyword argument and matches the call
already used by vLLM-Omni's generation runner. With that compatibility patch the
Orin run progresses through talker model loading and graph compilation.

The next reproducible blocker is:

```text
AttributeError: 'MemoryProfilingResult' object has no attribute 'total_consumed'
```

It occurs in `vllm_omni/worker/base.py` after the stage-0 profiling/warm-up run.
Installed vLLM 0.26 exposes `non_kv_cache_memory`, `torch_peak_increase`,
`non_torch_increase`, and `weights_memory`; current vLLM-Omni main reads the newer
`total_consumed` field. This is an upstream API-version mismatch, not an OOM: the
talker loaded 619 weights, used 1.91 GiB, selected `FLASH_ATTN` version 2, and
completed a 57.24-second first-run `torch.compile` pass before failing. Do not
blindly derive or inject the missing field in a robot deployment; pin a confirmed
vLLM/vLLM-Omni pair or carry a tested upstream compatibility fix first.

## Orin launch profile

`profiles/qwen3-tts-orin-nx.yaml` preserves upstream async PCM chunking and Code2Wav
correctness limits, but reduces `max_num_seqs` to 1 and uses conservative stage
reservations of 0.22 and 0.18 for 16 GB bring-up. These values are not a performance
claim. Once the compatibility gate above is resolved, start the isolated server with:

```bash
bash scripts/run-qwen3-tts-vllm-omni.sh
```

All paths can be overridden without editing the script:

```bash
VLLM_OMNI_RUNTIME_ROOT=/path/to/runtime \
QWEN3_TTS_MODEL=/path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice \
QWEN3_TTS_CUDSS_DIR=/path/to/libcudss/12 \
QWEN3_TTS_ALLOWED_MEDIA_PATH=/path/to/reference-audio-directory \
bash scripts/run-qwen3-tts-vllm-omni.sh
```

The speech service connects to
`ws://127.0.0.1:8091/v1/audio/speech/stream`, retains the WebSocket across sentences,
and starts ALSA playback on the first 24 kHz PCM chunk. The `tts.model` value sent
by the client must match the server's served model identifier; for a local model,
put the same absolute path in the selected host-local JSON profile.

## Voice-model semantics

The downloaded **CustomVoice** checkpoint provides preset speakers such as Ryan and
accepts style instructions. It does **not** clone an arbitrary reference recording.
Therefore the locally supplied `audio.wav` and its transcript are deliberately
not attached to this checkpoint.

The downloaded CustomVoice directory already contains its own `speech_tokenizer/`
weights. The separate `Qwen3-TTS-Tokenizer-12Hz` checkout is useful for standalone
codec work, but the vLLM-Omni CustomVoice launch does not need a second tokenizer
path in the speech-service configuration.

To clone that voice, download the matching `Qwen3-TTS-12Hz-0.6B-Base` checkpoint and
select `task_type: "Base"`, `reference_audio`, and the exact `reference_text` in the
JSON profile. Configuration validation rejects reference audio on CustomVoice so a
community user cannot unknowingly test a preset voice while believing it was cloned.

## Acceptance gate

Do not call this a production TTS baseline until a server run on the 16 GB Orin records:

- startup and peak unified-memory use without starving ASR or robot workloads;
- time to first playable PCM, audio RTF, underruns, and stop-to-silence latency;
- output continuity over multiple sentences on one WebSocket;
- transcript/style correctness and long-response stability;
- double-talk AEC and barge-in with the final speaker/microphone/enclosure.
