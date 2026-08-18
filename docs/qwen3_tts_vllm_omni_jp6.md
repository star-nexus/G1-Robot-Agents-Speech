# Qwen3-TTS with vLLM-Omni on JetPack 6.2

## Status

The speech service now contains a transport-neutral streaming TTS client, sentence
scheduler, ALSA player, render-reference tee, and hard barge-in path. The server
runtime remains isolated from the stable ASR environments.

The official **vLLM-Omni v0.26.0 release tag is now a runnable, measured
low-latency baseline** on JetPack 6.2. Both talker and Code2Wav stages start, the
official WebSocket endpoint becomes ready, and the speech service client receives
incremental 24 kHz PCM over one reusable connection. The selected Orin profile
uses `TRITON_ATTN` with Stage-0 `FULL_AND_PIECEWISE` CUDA Graphs; this avoids the
native SM87 FA2 full-graph capture stall and runs faster than real time in the
controlled streaming benchmark.

The JetPack 6.2 source-build experiment uses:

| Component | Selection |
|---|---|
| Platform | Jetson Orin NX 16 GB, SM 8.7, CUDA 12.6 |
| vLLM-Omni | official `v0.26.0` tag, commit `a4ea67a21b20054dacc6e83952f9bd407e8ee4e7` |
| vLLM | source commit `568afb3a1`, reported as `0.26.1.dev0`, source build for SM 8.7 |
| Python environment | separate host-local virtual environment |
| PyTorch | Jetson ARM64 2.11.0, CUDA 12.6 |
| Triton | 3.5.1 |
| torchaudio | matching 2.11.0 source build, isolated environment only |
| Model | local Qwen3-TTS-12Hz-0.6B-CustomVoice, preset voice Ryan |

Pinning the release is essential. The previously tested upstream `main` commit
`72a02b4` called APIs from a newer, unreleased vLLM line and failed against the
0.26 runtime at `MemoryProfilingResult.total_consumed`. vLLM-Omni is the
multistage orchestration and serving layer **on top of** vLLM, so a WebSocket
between mismatched copies would not repair that in-process Python API boundary.
The correct boundary is instead:

```text
speech service --WebSocket--> vLLM-Omni 0.26.0 --Python API--> vLLM 0.26.x
```

This combination is substantially newer than the project's sealed ASR PyTorch 2.5
environment. It must not be installed into `.venv-gpu`. Latest vLLM's ordinary
server wheels target a different CUDA/platform matrix; this experiment therefore
builds the CUDA extensions locally with `CUDA_HOME=/usr/local/cuda` and
`TORCH_CUDA_ARCH_LIST=8.7`.

## Reproducing the isolated build

Use a separate root and a Jetson-native PyTorch wheel. Check out the release in
its own directory; do not reuse a dirty `main` checkout:

```bash
git -C /path/to/vllm-omni fetch origin tag v0.26.0
git -C /path/to/vllm-omni worktree add /path/to/runtime/source-v0.26.0 v0.26.0
/path/to/runtime/.venv/bin/python -m pip install \
  --no-deps --no-build-isolation -e /path/to/runtime/source-v0.26.0
```

The launcher checks `vllm-omni==0.26.0` and vLLM `0.26.x` before loading either
model. The important vLLM source-build
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
unavailable optional Transformers feature before importing vLLM-Omni. vLLM's
model-independent warm-up also imports a MiniMax vision processor directly; a
minimal `InterpolationMode` shim prevents that import from loading binary system
vision operators in child processes. Set
`STAR_TTS_DISABLE_TORCHVISION=0` only after installing a Jetson-native
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

Do not apply the former `main`-branch `randomize_inputs` workaround to the release.
The v0.26.0 tag already uses the correct call and does not read
`MemoryProfilingResult.total_consumed`.

## Orin launch profile

`deploy/profiles/qwen3-tts-orin-nx.yaml` preserves upstream async PCM chunking and Code2Wav
correctness limits, reduces `max_num_seqs` to 1, and uses conservative stage
reservations of 0.23 and 0.18 for 16 GB. The talker limit is 1024 semantic tokens,
about 80 seconds at 12.5 Hz and far above a normal robot response. Stage 0 uses
`FULL_AND_PIECEWISE` graph mode for capture sizes 1 and 2. Stage 1 remains eager:
its separate Code2Wav graph path consumed substantially more reserved memory and
did not improve first-PCM latency in this robot workload. Start the isolated
server with:

```bash
bash scripts/run-qwen3-tts-vllm-omni.sh
```

All paths can be overridden without editing the script:

```bash
VLLM_OMNI_RUNTIME_ROOT=/path/to/runtime \
VLLM_OMNI_SOURCE_DIR=/path/to/runtime/source-v0.26.0 \
QWEN3_TTS_MODEL=/path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice \
QWEN3_TTS_CUDSS_DIR=/path/to/libcudss/12 \
QWEN3_TTS_ALLOWED_MEDIA_PATH=/path/to/reference-audio-directory \
bash scripts/run-qwen3-tts-vllm-omni.sh
```

The launcher defaults `QWEN3_TTS_ATTENTION_BACKEND=TRITON_ATTN`. It also forwards
extra vLLM arguments, so a diagnostic fallback can be requested without editing
the script. Do not select `FLASH_ATTN_2` with the default FULL graph profile on
this SM87 build; that is the reproduced stall combination.

The speech service connects to
`ws://127.0.0.1:8091/v1/audio/speech/stream`, retains the WebSocket across sentences,
and appends every 24 kHz PCM chunk to one bounded continuous playback timeline.
Generation and playback run on separate threads: the WebSocket reader may generate
subsequent sentences ahead of real-time playback, while a dedicated ALSA worker
consumes fixed 10/20 ms blocks. Sentence/request boundaries are not exposed to the
player, and backpressure reaches generation only when `audio_output.buffer_seconds`
is full. A barge-in atomically cancels generation, clears buffered PCM, and aborts
the hardware stream. The `tts.model` value sent by the client must match the
server's served model identifier; for a local model, put the same absolute path in
the selected host-local JSON profile.

### Robot-dialogue text chunking

The client does not equate a comma with a sentence. Its balanced robot policy is:

- complete (`is_final`) text keeps comma-separated clauses in one prosodic request;
- true sentence endings (`。.!?！？；;`) are safe low-latency boundaries while text
  is still streaming;
- commas, colons, enumeration commas, and dashes become candidates only after a
  minimum amount of speakable context exists;
- long text without punctuation is wrapped before it can delay speech indefinitely;
- a decimal point such as `3.14` is not treated as a sentence ending.

Length uses language-neutral approximate speech units rather than raw bytes: one
CJK/Kana/Hangul character is one unit and one Unicode word is 1.5 units. The model
profile exposes `tts.min_chunk_speech_units` (8),
`tts.preferred_chunk_speech_units` (14), and `tts.max_chunk_speech_units` (24).
`tts.max_buffer_characters` remains an absolute malformed-input safety limit. These
are model/dialogue policy fields in JSON, not hardware fields in `deploy.env`.

## Measured release validation

The first request after a clean launch incurred runtime JIT and is deliberately
reported rather than hidden. It produced 4.32 seconds of valid PCM in four chunks,
but first playable PCM took 103.09 seconds. The service logged JIT compilation for
multimodal RoPE, the code predictor, and Code2Wav kernels.

After warm-up, the same sentence produced 4.40 seconds of PCM with a 560.0 ms first
chunk and 8.72 s total generation time. A stronger persistent-connection test then
generated two utterances on one WebSocket:

| Utterance | First PCM | Total generation | Audio | Chunks |
|---|---:|---:|---:|---:|
| `Welcome to STAR.` | 568.8 ms | 3375.6 ms | 1.52 s | 2 |
| `How may I help you?` | 511.7 ms | 4736.6 ms | 2.32 s | 3 |

Those values establish the original eager functional baseline. A controlled
follow-up isolated the two graph layers and the attention backend. The selected
TRITON + Stage-0 FULL profile completed the formerly stalled FULL decode capture
in 2.08 seconds (all Stage-0 captures in about 5 seconds) and then completed six
steady streaming requests with 244.8 ms mean first PCM and 0.716 mean audio RTF.

The complete attribution, caveats, and comparison table are in
[Qwen3-TTS CUDA Graph attribution on Orin NX](qwen3_tts_cuda_graph_orin.md).
Machine-readable summaries and per-run responses are retained under
`benchmarks/orin_nx_2026-08-13_vllm_omni/experiments/`.

For the concurrency-one robot deployment, the production profile now explicitly
limits Stage-0 KV cache to 64 MiB instead of allowing vLLM to fill the general
serving memory budget. This saves about 2 GiB while retaining the same attention
and CUDA Graph baseline. The measurements, short-reply token envelope, native
runtime lower bound, rollback profile, and stopping decisions are documented in
[Qwen3-TTS single-user memory profile on Jetson Orin NX](qwen3_tts_memory_orin_nx.md).

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

Do not call this a production TTS performance baseline until a server run on the
16 GB Orin records:

- startup and peak unified-memory use without starving ASR or robot workloads;
- time to first playable PCM, audio RTF, underruns, and stop-to-silence latency;
- output continuity over multiple sentences on one WebSocket;
- transcript/style correctness and long-response stability;
- double-talk AEC and barge-in with the final speaker/microphone/enclosure.
