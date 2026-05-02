#!/usr/bin/env bash
set -euo pipefail

# ===== Default config =====
MODEL_NAME="${MODEL_NAME:-/mnt/d/Labs/Efficient-LLM-Inference/models/pythia-70m}"
DATASETS="${DATASETS:-pg19 wikitext}"
METHODS="${METHODS:-streaming_prefill streaming_online streaming_decode_only knorm_prefill knorm_online snapkv_prefill pyramidkv_prefill}"
BUDGETS="${BUDGETS:-256 512 768 1024}"
TOKEN_OFFSETS="${TOKEN_OFFSETS:-0}"

CONTEXT_LEN="${CONTEXT_LEN:-1536}"
TARGET_LEN="${TARGET_LEN:-512}"
N_SINK="${N_SINK:-4}"
POSITION_MODE="${POSITION_MODE:-absolute}"

OUTPUT_CSV="${OUTPUT_CSV:-results/all_methods.csv}"
OVERWRITE="${OVERWRITE:-1}"

# ===== Prepare output =====
mkdir -p "$(dirname "$OUTPUT_CSV")"

if [ "$OVERWRITE" = "1" ]; then
  rm -f "$OUTPUT_CSV"
fi

echo "================================================================================"
echo "Running all KV compression evaluations"
echo "MODEL_NAME:     $MODEL_NAME"
echo "DATASETS:       $DATASETS"
echo "METHODS:        baseline $METHODS"
echo "BUDGETS:        $BUDGETS"
echo "TOKEN_OFFSETS:  $TOKEN_OFFSETS"
echo "CONTEXT_LEN:    $CONTEXT_LEN"
echo "TARGET_LEN:     $TARGET_LEN"
echo "N_SINK:         $N_SINK"
echo "POSITION_MODE:  $POSITION_MODE"
echo "OUTPUT_CSV:     $OUTPUT_CSV"
echo "OVERWRITE:      $OVERWRITE"
echo "================================================================================"

# ===== Run baseline once per dataset/offset =====
for DATASET in $DATASETS; do
  for OFFSET in $TOKEN_OFFSETS; do
    echo
    echo "--------------------------------------------------------------------------------"
    echo "[baseline] dataset=$DATASET offset=$OFFSET"
    echo "--------------------------------------------------------------------------------"

    python run_eval.py \
      --model_name "$MODEL_NAME" \
      --dataset "$DATASET" \
      --method baseline \
      --context_len "$CONTEXT_LEN" \
      --target_len "$TARGET_LEN" \
      --budget 0 \
      --n_sink "$N_SINK" \
      --token_offset "$OFFSET" \
      --position_mode "$POSITION_MODE" \
      --output_csv "$OUTPUT_CSV"
  done
done

# ===== Run all compression methods =====
for DATASET in $DATASETS; do
  for OFFSET in $TOKEN_OFFSETS; do
    for METHOD in $METHODS; do
      for BUDGET in $BUDGETS; do
        echo
        echo "--------------------------------------------------------------------------------"
        echo "[$METHOD] dataset=$DATASET budget=$BUDGET offset=$OFFSET"
        echo "--------------------------------------------------------------------------------"

        python run_eval.py \
          --model_name "$MODEL_NAME" \
          --dataset "$DATASET" \
          --method "$METHOD" \
          --context_len "$CONTEXT_LEN" \
          --target_len "$TARGET_LEN" \
          --budget "$BUDGET" \
          --n_sink "$N_SINK" \
          --token_offset "$OFFSET" \
          --position_mode "$POSITION_MODE" \
          --output_csv "$OUTPUT_CSV"
      done
    done
  done
done


python run_eval.py \
  --dataset pg19 \
  --method think_prefill \
  --context_len 1536 \
  --target_len 512 \
  --budget 0 \
  --output_csv results/all_methods.csv

python run_eval.py \
  --dataset wikitext \
  --method think_prefill \
  --context_len 1536 \
  --target_len 512 \
  --budget 0 \
  --output_csv results/all_methods.csv

echo
echo "================================================================================"
echo "All evaluations finished."
echo "CSV saved to: $OUTPUT_CSV"
echo "================================================================================"

python - <<PY
import pandas as pd
path = "$OUTPUT_CSV"
df = pd.read_csv(path)
cols = [
    "dataset", "method", "budget", "token_offset",
    "ppl", "tpot_ms", "throughput_tok_s", "peak_mem_mb",
    "prefill_cache_len_avg", "final_cache_len_avg",
    "prefill_cache_lens", "final_cache_lens",
]
cols = [c for c in cols if c in df.columns]
print(df[cols].to_string(index=False))
PY
