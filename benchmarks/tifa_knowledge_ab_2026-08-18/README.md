# Tifa role-knowledge A/B on Jetson Orin NX

Date: 2026-08-18

This small interactive test compares the same Tifa Role Package with three
knowledge configurations:

1. `provider: none` with no knowledge paths;
2. `provider: none` while retaining the unused `core` and `cards` paths;
3. `provider: keyword` with the bundled core lore and keyword cards.

The raw per-turn timings are in `per_run.csv`. The keyword cold request and the
unmatched extra Aerith question are marked and excluded from the steady-state
eight-intent comparison.

| Variant | Mean first text | Median first text | Mean total | Median total | Mean chunks |
| --- | ---: | ---: | ---: | ---: | ---: |
| None, minimal | 395.5 ms | 410.2 ms | 2,114.8 ms | 2,185.6 ms | 20.0 |
| None, paths retained | 393.4 ms | 387.8 ms | 2,038.1 ms | 2,134.1 ms | 19.9 |
| Keyword, warm | 618.0 ms | 585.4 ms | 2,112.6 ms | 1,785.0 ms | 16.4 |

Keeping paths under `provider: none` has no runtime effect. Keyword knowledge
adds about 224 ms to mean first-text latency in this sample, but it also grounds
facts and shortens several answers, leaving mean request-total time nearly
unchanged. The decoder's approximate milliseconds per streamed text chunk is
similar across all three variants.

This is an exploratory product test, not a controlled performance benchmark:
each intent ran once per variant, the three-turn window accumulated different
answers, one Nibelheim ASR string differed, and llama.cpp slot/cache state was
not reset between conditions.
