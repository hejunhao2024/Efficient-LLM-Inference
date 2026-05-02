# Results

## Experimental Setup

- **Model:** Pythia-70M
- **Datasets:** PG-19 single long-text sample and WikiText-2 test split
- **Context length:** 1536 tokens
- **Target length:** 512 tokens
- **Position mode:** absolute position ids after KV compression
- **PPL protocol:** teacher-forced continuation perplexity
- **Reported runtime metrics:** prefill time / TTFT proxy, TPOT, throughput, peak VRAM, and cache-length statistics

`ttft_s` is best interpreted as **prefill time / TTFT proxy**, because PPL evaluation uses teacher-forced decoding instead of open-ended generation.

## Main Results at Budget 512

The following table compares the main methods at budget 512. Baseline and ThinK do not use a token budget.

| Dataset   | Method            | Budget   |   PPL ↓ |   ΔPPL vs base |   TPOT ms ↓ |   Throughput tok/s ↑ |   Peak VRAM MB ↓ |   VRAM Δ MB |   Prefill Cache Avg |   Final Cache Avg |
|:----------|:------------------|:---------|--------:|---------------:|------------:|---------------------:|-----------------:|------------:|--------------------:|------------------:|
| pg19      | baseline          | -        |   31.4  |           0    |        7.19 |                139.1 |            318.3 |         0   |                1536 |              2047 |
| pg19      | streaming_prefill | 512      |   30.26 |          -1.15 |        7.31 |                136.7 |            305.3 |       -13   |                 512 |              1023 |
| pg19      | streaming_online  | 512      |   30.41 |          -0.99 |        7.77 |                128.7 |            299.7 |       -18.6 |                 512 |               512 |
| pg19      | knorm_prefill     | 512      |   35.04 |           3.64 |        7.14 |                140   |            305.3 |       -13   |                 512 |              1023 |
| pg19      | snapkv_prefill    | 512      |   29.57 |          -1.84 |        6.11 |                163.7 |            305.3 |       -13   |                 512 |              1023 |
| pg19      | pyramidkv_prefill | 512      |   30.33 |          -1.08 |        7.56 |                132.3 |            308.1 |       -10.3 |                 512 |              1023 |
| pg19      | think_prefill     | -        |   47.05 |          15.64 |        7.13 |                140.3 |            318.3 |         0   |                1536 |              2047 |
| wikitext  | baseline          | -        |   59.78 |           0    |        6.75 |                148.2 |            318.3 |         0   |                1536 |              2047 |
| wikitext  | streaming_prefill | 512      |   60.16 |           0.38 |        6.76 |                148   |            305.3 |       -13   |                 512 |              1023 |
| wikitext  | streaming_online  | 512      |   62.91 |           3.14 |        8.13 |                123.1 |            299.7 |       -18.6 |                 512 |               512 |
| wikitext  | knorm_prefill     | 512      |   69.81 |          10.03 |        7.06 |                141.7 |            305.3 |       -13   |                 512 |              1023 |
| wikitext  | snapkv_prefill    | 512      |   60.47 |           0.69 |        7.34 |                136.2 |            305.3 |       -13   |                 512 |              1023 |
| wikitext  | pyramidkv_prefill | 512      |   61.86 |           2.09 |        7.4  |                135.1 |            308.1 |       -10.3 |                 512 |              1023 |
| wikitext  | think_prefill     | -        |  111.92 |          52.14 |        6.13 |                163.1 |            318.3 |         0   |                1536 |              2047 |

## Best PPL per Main Method

For each method with multiple budgets, this table reports the best PPL among budgets `256 / 512 / 768 / 1024`.

| Dataset   | Method            | Best Budget   |   Best PPL ↓ |   TPOT ms |   Peak VRAM MB |   Final Cache Avg |
|:----------|:------------------|:--------------|-------------:|----------:|---------------:|------------------:|
| pg19      | baseline          | -             |        31.4  |      7.19 |          318.3 |              2047 |
| pg19      | streaming_prefill | 256           |        29.95 |      7.8  |          302.3 |               767 |
| pg19      | streaming_online  | 512           |        30.41 |      7.77 |          299.7 |               512 |
| pg19      | knorm_prefill     | 1024          |        32.33 |      7.52 |          311.8 |              1535 |
| pg19      | snapkv_prefill    | 512           |        29.57 |      6.11 |          305.3 |              1023 |
| pg19      | pyramidkv_prefill | 256           |        29.88 |      9.1  |          302.3 |               767 |
| wikitext  | baseline          | -             |        59.78 |      6.75 |          318.3 |              2047 |
| wikitext  | streaming_prefill | 1024          |        59.23 |      7.73 |          311.8 |              1535 |
| wikitext  | streaming_online  | 1024          |        59.56 |      7.73 |          306.3 |              1024 |
| wikitext  | knorm_prefill     | 1024          |        65.76 |      6.91 |          311.8 |              1535 |
| wikitext  | snapkv_prefill    | 1024          |        59.39 |      7.63 |          311.8 |              1535 |
| wikitext  | pyramidkv_prefill | 768           |        59.54 |      9.74 |          313   |              1279 |

## Additional Ablation Results

These methods are useful for understanding behavior, but they are not used as the main comparison.

| Dataset   | Method                | Budget   |   PPL ↓ |   TPOT ms ↓ |   Peak VRAM MB ↓ |   Prefill Cache Avg |   Final Cache Avg |
|:----------|:----------------------|:---------|--------:|------------:|-----------------:|--------------------:|------------------:|
| pg19      | streaming_decode_only | 512      |   30.39 |        8.05 |            312.7 |                1536 |               512 |
| pg19      | knorm_online          | 512      |  309.24 |        8.88 |            299.7 |                 512 |               512 |
| pg19      | think_prefill         | -        |   47.05 |        7.13 |            318.3 |                1536 |              2047 |
| wikitext  | streaming_decode_only | 512      |   61.93 |        7.43 |            312.7 |                1536 |               512 |
| wikitext  | knorm_online          | 512      |  504.82 |        9.19 |            299.7 |                 512 |               512 |
| wikitext  | think_prefill         | -        |  111.92 |        6.13 |            318.3 |                1536 |              2047 |

## Analysis

### Overall behavior

The implemented KV compression framework behaves as expected. Baseline keeps the full cache, with an average prefill cache length of 1536 and final cache length of 2047. Prefill-only token pruning methods first reduce the context cache to the target budget and then append target tokens during teacher-forced decoding. Online methods keep the final cache length fixed at the budget.

### StreamingLLM

StreamingLLM is stable across both datasets. In `streaming_prefill`, the PPL stays close to baseline while reducing the prefill cache length from 1536 to the selected budget. `streaming_online` further keeps the final cache length fixed, leading to lower peak memory, but can hurt PPL more on WikiText at small budgets.

### SnapKV

SnapKV is one of the strongest methods in this experiment. At budget 512, it achieves lower PPL than baseline on PG-19 and remains close to baseline on WikiText. This suggests that attention-based selection from the recent observation window is more effective than simple norm-based pruning.

### PyramidKV

PyramidKV correctly implements layer-wise budget allocation. Lower layers keep more tokens and higher layers keep fewer tokens, while the average layer budget matches the target budget. Its PPL is competitive with SnapKV and StreamingLLM, although the best budget differs by dataset.

### KNorm

KNorm prefill is a useful heuristic baseline, but it is consistently weaker than SnapKV and StreamingLLM. Its PPL improves as the budget increases, which indicates that the implementation is behaving reasonably.

### KNorm online

`knorm_online` is unstable and significantly degrades PPL. This is likely because it repeatedly reselects cache entries during decoding without explicitly preserving local recency. It is better treated as a failure case / ablation rather than a main method.

### ThinK

ThinK is implemented as key-channel masking. It zeros out low-importance key dimensions but keeps the dense KV tensor shape unchanged. Therefore, it affects PPL but does not reduce actual KV cache memory in this implementation. The results show clear PPL degradation, especially on WikiText, so it should be treated as a channel-wise ablation rather than a practical memory-saving result.

### Latency and memory

Peak VRAM generally decreases when the cache length is reduced. However, latency results are noisy because Pythia-70M is small and the implementation uses Python hooks plus tensor slicing/gather operations. Therefore, PPL, cache-length statistics, and peak memory are more reliable indicators than raw TPOT speedup in this small-model setting.

## Key Takeaways

- SnapKV and StreamingLLM provide the best trade-off between PPL and cache reduction.
- PyramidKV successfully demonstrates layer-wise adaptive cache allocation.
- KNorm is simple but weaker than attention-based methods.
- Online KNorm is not suitable for stable decoding in this implementation.
- ThinK masking is useful as an ablation but does not reduce memory without a compressed attention kernel.
