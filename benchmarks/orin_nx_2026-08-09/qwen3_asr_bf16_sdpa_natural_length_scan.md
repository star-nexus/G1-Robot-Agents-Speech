# Qwen3-ASR BF16 SDPA provided-audio length scan

Generated: 2026-08-09T20:52:14.973193+08:00

| Case | Audio (s) | Mean (ms) | Median (ms) | P95 (ms) | RTF | Tokens | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| test_audio_6s | 6.079 | 1040.0 | 1040.6 | 1042.0 | 0.1711 | 12.0 | 11.65 |
| test_audio_30s | 30.313 | 7983.8 | 7982.7 | 8003.5 | 0.2634 | 98.0 | 12.31 |
| test_audio_60s | 60.436 | 16118.6 | 16125.2 | 16138.1 | 0.2667 | 202.0 | 12.57 |
| test_audio_120s | 120.425 | 20415.7 | 20427.0 | 20428.3 | 0.1695 | 256.0 | 12.60 |

Times are synchronized wall-clock measurements. Tokens/s is generated tokens divided by `generate_ms`.
