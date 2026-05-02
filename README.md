# Efficient-LLM-Inference —— NLP Homework

A lightweight KV cache compression benchmark framework for **Pythia-70M**.  
This project implements several training-free KV cache optimization methods and evaluates them on PG-19 and WikiText using perplexity, latency, throughput, memory, and cache-length statistics.

## Implemented Methods

The current implementation supports:

- **Baseline**: no KV cache compression
- **StreamingLLM**
  - `streaming_prefill`: compress after prefill only
  - `streaming_online`: compress after prefill and maintain a fixed-size cache during decoding
  - `streaming_decode_only`: keep full prefill cache, then apply sink+recent eviction during decoding
- **KNorm**
  - `knorm_prefill`
  - `knorm_online`
- **SnapKV**
  - `snapkv_prefill`
- **PyramidKV**
  - `pyramidkv_prefill`
- **ThinK**
  - `think_prefill`

Notes:

- SnapKV and PyramidKV are implemented as **prefill-only** KV compression methods.
- StreamingLLM supports both prefill-only and online cache maintenance.
- ThinK is implemented as a **key-channel masking** method. It zeros out low-importance key channels but keeps the dense KV tensor shape unchanged, so it does **not** reduce actual memory in this implementation.
- Key rerotation / RoPE position remapping is not implemented. All main experiments use absolute position ids.

## Model

The default model is:

```text
EleutherAI/pythia-70m
```

The experiments are designed around Pythia-70M / GPT-NeoX style attention and HuggingFace `DynamicCache`.

## Dataset and Model Download

Download the model and datasets:

```bash
hf download EleutherAI/pythia-70m --local-dir ./pythia-70m
python download_dataset.py
```

Expected dataset paths:

```text
data/PG-19/Reminiscences_of_Pioneer_Days_in_St._Paul_by_Frank_Moore.txt
data/WikiText-2/test.txt
```

## Installation

```bash
pip install -e .
```

If imports are not resolved, run commands from the project root or set:

```bash
export PYTHONPATH=.
```

## Evaluation Protocol

The main evaluation uses:

```text
context_len = 1536
target_len  = 512
total tokens = 2048
```

For each sample window:

1. The first `context_len` tokens are used for context prefill.
2. KV cache compression is applied depending on the selected method.
3. The next `target_len` tokens are evaluated with teacher forcing.
4. At step `i`, the model receives the ground-truth token `target[i]` and predicts `target[i+1]`.
5. Perplexity is computed as:

```text
PPL = exp(mean cross entropy)
```

The evaluation does **not** use `model.generate()` for PPL.

### Position IDs

For compressed KV cache methods, decode-time position ids are manually set to the original absolute positions:

```text
position_id = context_len + step
```

This keeps the retained keys and new queries in the same RoPE coordinate system.  
The implementation does not perform key rerotation.

## Metrics

The default benchmark reports:

- **PPL**: teacher-forced continuation perplexity
- **TTFT / prefill_time_s**: context prefill time, including prefill compression if used
- **TPOT**: average time per target decoding step
- **Throughput**: target tokens per second
- **Peak VRAM**: measured with `torch.cuda.max_memory_allocated`
- **Cache length statistics**
  - `prefill_cache_len_avg`
  - `final_cache_len_avg`
  - `prefill_cache_lens`
  - `final_cache_lens`

Note: `ttft_s` in the current implementation is best interpreted as **prefill time / TTFT proxy**, because PPL evaluation uses teacher-forced decoding instead of open-ended generation.



## Run All Experiments

```bash
bash scripts/run_all_methods.sh
```

The default script evaluates PG-19 and WikiText across multiple methods and budgets.

To run a quick debug experiment:

```bash
DATASETS="pg19" \
BUDGETS="32" \
CONTEXT_LEN=64 \
TARGET_LEN=32 \
METHODS="streaming_prefill knorm_prefill snapkv_prefill pyramidkv_prefill" \
OUTPUT_CSV="results/debug_all_methods.csv" \
bash scripts/run_all_methods.sh
```

To run the main configuration:

```bash
DATASETS="pg19 wikitext" \
BUDGETS="256 512 768 1024" \
CONTEXT_LEN=1536 \
TARGET_LEN=512 \
OUTPUT_CSV="results/all_methods.csv" \
bash scripts/run_all_methods.sh
```

## Results

Full results are saved in:

```text
results/all_methods.csv
```

The CSV includes PPL, latency, throughput, peak memory, and layer-wise cache-length statistics.


See detailed results and analysis in [results/result.md](results/result.md).

## Main Observations

From the current experiments:

- SnapKV and StreamingLLM maintain PPL close to the baseline while reducing KV cache length.
- PyramidKV correctly allocates larger budgets to lower layers and smaller budgets to higher layers.
- KNorm prefill works as a simple heuristic baseline, but is generally weaker than SnapKV.
- KNorm online is unstable and can significantly degrade PPL, showing that naive dynamic norm-based eviction is not suitable for decoding.
- ThinK channel masking affects PPL but does not reduce memory in this dense-cache implementation.

## Project Structure

```text
.
├── pythia_kvpress
│   ├── __init__.py
│   ├── hooks.py                 # attention hook registration
│   ├── cache_utils.py           # DynamicCache read/write helpers
│   ├── presses
│   │   ├── __init__.py
│   │   ├── base.py              # BasePress
│   │   ├── scorer.py            # score-based pruning base class
│   │   ├── streaming.py         # StreamingLLM
│   │   ├── knorm.py             # KNorm
│   │   ├── snapkv.py            # SnapKV
│   │   ├── pyramidkv.py         # PyramidKV
│   │   └── think.py             # ThinK-inspired key-channel masking
│   └── eval
│       ├── __init__.py
│       └── ppl.py               # teacher-forced PPL evaluation
├── run_eval.py                  # single experiment entry point
├── scripts
│   └── run_all_methods.sh       # full benchmark script
├── tests                        # unit and visualization tests
└── results                      # CSV results and summaries
```
