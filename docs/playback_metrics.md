# Playback metric semantics

`AlsaOutputPlayer.metrics()` returns a lock-consistent snapshot. Counters are
monotonic for one player lifetime; `last_*` values describe the latest event and
`max_*` values retain the lifetime maximum.

The hardware commit point is the final generation check in the playback thread.
Passing that check mints one monotonic integer commit token bound to the checked
generation. The exact pair is required at the only function that can feed the AEC
render reference and call `RawOutputStream.write()`. This adds no per-block object
or container allocation to the 10 ms path.

| Metric | Meaning and update path |
|---|---|
| `audio_output_device` | Stable PortAudio device name resolved when `start()` succeeds. |
| `audio_output_latency_ms` | PortAudio's reported stream latency, sampled during `start()`. |
| `audio_output_blocks` | Queued speaker-PCM blocks whose `RawOutputStream.write()` returned successfully. WebRTC clock-silence writes are excluded. |
| `audio_output_bytes` | Bytes in those successfully returned queued speaker-PCM writes. WebRTC clock silence is excluded. |
| `audio_output_continuous_render_clock` | Whether this player uses its persistent speaker writer as a continuous WebRTC render clock. This is reachable for WebRTC processing with the persistent interrupt strategy. |
| `audio_output_render_clock_silence_blocks` | Successful all-zero speaker/AEC writes made only to advance the continuous 10 ms render clock while no queued PCM was available. These blocks are not Turn/Epoch work and cannot count as stale or residual audible PCM. |
| `audio_output_interruptions` | Calls to `abort()` that advance the player generation. |
| `audio_output_buffered_ms` | Snapshot of ring samples plus the currently dequeued block. |
| `audio_output_peak_buffered_ms` | Maximum ring occupancy, updated after successful enqueue writes. |
| `audio_output_buffer_capacity_ms` | Fixed configured ring capacity. |
| `audio_output_backpressure_waits` | Producer waits caused by a full ring. |
| `audio_output_playback_errors` | Uncancelled playback-loop exceptions from restart or hardware submission. |
| `audio_output_interrupt_strategy` | Configured `persistent` or rollback `hard_abort` strategy. |
| `audio_output_stream_active` | Current PortAudio `active` property, or `null` without a stream. |
| `audio_output_playback_thread_alive` | Whether the dedicated playback thread is alive at snapshot time. |
| `audio_output_stream_abort_calls` | Actual PortAudio abort calls from hard interruption or shutdown fallback. |
| `audio_output_stream_abort_errors` | Exceptions raised by those PortAudio abort calls. |
| `audio_output_close_abort_fallbacks` | Persistent-stream shutdowns that did not stop gracefully in 250 ms and required abort. |
| `audio_output_stream_restart_attempts` | Inactive-stream restart calls made before the final generation gate. |
| `audio_output_stream_restart_successes` | Restart calls that returned successfully. |
| `audio_output_stream_restart_errors` | Restart calls that raised an exception. |
| `audio_output_stream_active_transitions` | Observed changes after the initial stream-active observation. |
| `audio_output_write_underflows` | Successful writes for which sounddevice returned its PortAudio underflow flag. |
| `audio_output_write_errors` | Uncancelled exceptions in the restart/submission portion of the playback loop. |
| `audio_output_hardware_write_in_progress` | Whether a committed block is currently in the AEC/PortAudio submission section. |
| `audio_output_hardware_write_generation` | Player generation of that in-progress committed block, otherwise `null`. |
| `audio_output_hardware_write_kind` | `pcm` for queued speaker PCM, `aec_silence` for an unscoped render-clock silence block, otherwise `null`. |
| `audio_output_hardware_write_in_progress_ms` | Elapsed time since that block crossed the final gate, otherwise `null`. |
| `audio_output_stale_blocks_dropped_before_write` | Dequeued blocks rejected by a generation/stop check before a commit token is minted. They reach neither AEC nor PortAudio. |
| `audio_output_forbidden_stale_write_attempts` | Blocks presented to the hardware-submission function without the exact active commit token. Each attempt is logged and rejected before both AEC and PortAudio. This replaces the unimplemented `audio_output_post_invalidation_stale_write_submissions`. |
| `audio_output_residual_writes_committed` | Interruptions that find a block already holding a valid commit token. The block is bounded residual audio, not a violation. |
| `audio_output_residual_write_completions` | Those committed writes whose PortAudio call returns after their generation advances. |
| `audio_output_recovery_pending` | An interruption has occurred and no replacement-generation write has yet completed. |
| `audio_output_recoveries` | First successful replacement-generation write after each interruption. |
| `audio_output_replacement_pcm_arrivals` | First post-interrupt provider PCM-ready notification. |
| `audio_output_replacement_enqueues` | First post-interrupt call that offers replacement PCM to the player. |
| `audio_output_replacement_dequeues` | First post-interrupt replacement block removed from the ring. |
| `audio_output_first_replacement_pcm_ms` | Latest interrupt-to-provider-PCM-ready latency. |
| `audio_output_first_replacement_enqueue_ms` | Latest interrupt-to-player-enqueue latency. |
| `audio_output_first_replacement_dequeue_ms` | Latest interrupt-to-playback-thread-dequeue latency. |
| `audio_output_last_write_ms` | Duration of the latest successful PortAudio write call. |
| `audio_output_max_write_ms` | Maximum successful PortAudio write duration. |
| `audio_output_last_abort_ms` | Duration of the latest hard-interrupt PortAudio abort; `null` when none occurred. |
| `audio_output_max_abort_ms` | Maximum hard-interrupt PortAudio abort duration. |
| `audio_output_last_recovery_ms` | Latest interruption-to-first-successful-replacement-write latency. |
| `audio_output_max_recovery_ms` | Maximum interruption-to-first-successful-replacement-write latency. |

The removed metric could only report its initialized zero. The replacement does
not claim to observe an impossible write retrospectively: it instruments the sole
hardware-submission boundary and proves that any caller lacking the commit minted
by the final gate is detected and prevented. Invalidation after token issuance does
not revoke that one token; it is accounted for by the two residual-write metrics.
