# Runtime Timing Audit v1

The integrated Runtime emits one `RUNTIME_TURN_TIMELINE` JSON record for every
terminal `Session / Turn / Epoch`. The audit is observational: it does not alter
VAD, model, sentence scheduling, cancellation, PCM buffering, or playback.

All stage timestamps use `time.monotonic_ns()` from the same host. Acoustic
`speech_start`, `speech_end`, and `vad_ready` are attached retrospectively when
ASR assigns the canonical Turn ID; their values come directly from the VAD
utterance and are not reconstructed from wall-clock logs. `logged_unix_ns` is
only for correlating the completed record with other terminal logs.

## Timeline stages

| Stage | Exact meaning |
|---|---|
| `speech_start` / `speech_end` | VAD utterance acoustic bounds. |
| `vad_speech_edge` | Early sherpa VAD `False -> True` edge on the previous active response; the primary acoustic barge-in trigger. |
| `vad_ready` | VAD emitted the completed utterance after its tail decision. |
| `asr_start` / `asr_final` | ASR provider call bounds. |
| `turn_started` | The Control Plane installed the Turn's initial epoch. |
| `speech_event_published` | Final speech was accepted by the selected transport. |
| `agent_start` | Agent processing began for the exact stamp. |
| `agent_first_delta` | First non-empty provider delta arrived; this is not terminal display time. |
| `agent_first_delta_published` | First delta passed the bridge look-ahead and reached the voice transport. |
| `tts_segment_ready` | Streaming text first formed a synthesizable sentence/clause. |
| `tts_segment_enqueued` / `tts_synthesis_start` | First TTS request queue and worker boundaries. |
| `playback_active` | TTS published the active playback lifecycle edge. |
| `tts_first_pcm` | TTS provider yielded its first PCM bytes. |
| `player_enqueue_begin` / `player_enqueue_complete` | First PCM append call bounds. |
| `playback_dequeue` | Playback removed the first tagged samples from the ring. |
| `playback_prewrite_commit` | Final validity gate minted the hardware write commit token. |
| `hardware_write_begin` / `hardware_write_complete` | Actual PortAudio `stream.write()` call bounds. |
| `agent_complete` | Full answer was generated and committed to Agent memory. |
| `playback_complete` | Final response PCM timeline drained. |

PCM provenance uses run-length spans parallel to the sample ring. It adds one
small metadata span per enqueue run, not per sample, and does not copy or
serialize PCM. A hardware block joining adjacent spans marks the first write for
every represented exact stamp.

Cancellation records can additionally contain `invalidated`,
`agent_cancel_requested`, `tts_interrupt`, `playback_abort`, `superseded`, or
provider error stages. A turn emits once with an outcome such as `completed`,
`completed_with_errors`, `invalidated`, or `superseded`.

`asr_final_fallback_cleanup` is an optional control marker on the previous stamp.
It proves that a newly authoritative Turn caused output-only cleanup before its
SpeechEvent was published. The cleanup does not call `RuntimeControlPlane.invalidate`
and therefore cannot advance the new Turn's epoch. Its `output_interrupted` field
distinguishes a real stale-output flush from an idempotent no-op after an earlier
VAD barge-in.

`vad_output_preempted` is recorded after the VAD path has advanced the persistent
player generation and cleared local TTS/PCM work. It is separate from the generic,
first-observation `playback_abort` event so an earlier queue-full or explicit abort
cannot produce a misleading negative VAD cutoff duration.

`timeline[].at_ms` is relative to acoustic `speech_start` when available;
`since_previous_ms` is relative to the preceding observed event. The
`durations_ms` object contains only intervals whose endpoints were observed.
`missing_events` is intentional evidence: the audit never fills a missing stage
with a guessed timestamp. `origin_monotonic_ns` preserves exact cross-Turn
correlation on the same boot: an event's absolute monotonic time is the origin
plus `at_ms`.

The complete record is enabled by the integrated in-process composition, where
ASR, Agent, TTS, and playback share one audit instance. Split DDS/ROS 2
deployments retain their control metadata but do not pretend that process-local
partial observations form one complete timeline in this version.
