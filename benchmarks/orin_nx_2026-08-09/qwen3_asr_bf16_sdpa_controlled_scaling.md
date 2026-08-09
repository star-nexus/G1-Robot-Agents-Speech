# Qwen3-ASR BF16 SDPA controlled audio-length scaling

Generated: 2026-08-09T20:41:41.953248+08:00

| Case | Audio (s) | Mean (ms) | Median (ms) | P95 (ms) | RTF | Tokens | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x | 5.592 | 1257.9 | 1256.5 | 1267.1 | 0.2249 | 15.0 | 12.03 |
| 2x | 11.184 | 2155.8 | 2155.6 | 2168.0 | 0.1928 | 26.0 | 12.13 |
| 4x | 22.368 | 3955.8 | 3949.1 | 4037.1 | 0.1768 | 48.0 | 12.22 |
| 8x | 44.736 | 6158.3 | 6201.9 | 6252.3 | 0.1377 | 74.0 | 12.10 |

Times are synchronized wall-clock measurements. Tokens/s is generated tokens divided by `generate_ms`.
