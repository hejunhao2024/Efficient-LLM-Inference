import argparse
import contextlib
import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from pythia_kvpress.eval import continuation_ppl
from pythia_kvpress.presses import (
    StreamingLLMPress,
    KNormPress,
    SnapKVPress,
    PyramidKVPress,
    ThinKPress,
)


DEFAULT_DATASET_PATHS = {
    "pg19": "data/PG-19/Reminiscences_of_Pioneer_Days_in_St._Paul_by_Frank_Moore.txt",
    "wikitext": "data/WikiText-2/test.txt",
}


ALL_METHODS = [
    "baseline",
    "streaming_prefill",
    "streaming_online",
    "streaming_decode_only",
    "knorm_prefill",
    "knorm_online",
    "snapkv_prefill",
    "pyramidkv_prefill",
    "think_prefill",
]


def load_text(dataset: str, dataset_path: str | None = None) -> str:
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_PATHS.get(dataset)

    if dataset_path is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Please pass --dataset_path explicitly."
        )

    return path.read_text(encoding="utf-8")


def tokenize_window(tokenizer, text: str, total_len: int, token_offset: int = 0):
    all_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids

    if all_ids.shape[1] < token_offset + total_len:
        raise ValueError(
            f"Not enough tokens. Got {all_ids.shape[1]}, "
            f"need token_offset + total_len = {token_offset + total_len}."
        )

    return all_ids[:, token_offset : token_offset + total_len]


def build_presses(method: str, budget: int, n_sink: int):
    """
    Returns:
        prefill_press, decoding_press

    Naming:
      *_prefill:
          compress only after context prefill

      *_online:
          compress after prefill and keep compressing during decode

      *_decode_only:
          do not compress prefill, only start compression during decode
    """
    if method == "baseline":
        return None, None

    if method == "streaming_prefill":
        return (
            StreamingLLMPress(mode="prefill", budget=budget, n_sink=n_sink),
            None,
        )

    if method == "streaming_online":
        return (
            StreamingLLMPress(mode="prefill", budget=budget, n_sink=n_sink),
            StreamingLLMPress(mode="decode", budget=budget, n_sink=n_sink),
        )

    if method == "streaming_decode_only":
        return (
            None,
            StreamingLLMPress(mode="decode", budget=budget, n_sink=n_sink),
        )

    if method == "knorm_prefill":
        return (
            KNormPress(mode="prefill", budget=budget, keep_low_norm=True),
            None,
        )

    if method == "knorm_online":
        return (
            KNormPress(mode="prefill", budget=budget, keep_low_norm=True),
            KNormPress(mode="decode", budget=budget, keep_low_norm=True),
        )

    if method == "snapkv_prefill":
        return (
            SnapKVPress(
                mode="prefill",
                budget=budget,
                window_size=32,
                kernel_size=7,
            ),
            None,
        )

    if method == "pyramidkv_prefill":
        return (
            PyramidKVPress(
                mode="prefill",
                budget=budget,
                window_size=32,
                kernel_size=7,
                beta=20,
            ),
            None,
        )
    
    if method == "think_prefill":
        return (
            ThinKPress(
                mode="prefill",
                key_channel_compression_ratio=0.5,
                window_size=32,
            ),
            None,
        )

    raise ValueError(f"Unknown method: {method}")


def cache_length_stats(cache, num_layers: int) -> dict:
    """
    Return layer-wise cache length statistics.

    This matters for PyramidKV because each layer may keep a different number
    of KV tokens.
    """
    lengths = [int(cache.get_seq_length(i)) for i in range(num_layers)]

    return {
        "layer0": lengths[0],
        "avg": sum(lengths) / len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "all": lengths,
    }


@torch.no_grad()
def prefill_context(model, context_ids, prefill_press=None):
    cache = DynamicCache(config=model.config)

    if prefill_press is None:
        outputs = model(
            input_ids=context_ids,
            past_key_values=cache,
            use_cache=True,
        )
    else:
        with prefill_press(model):
            outputs = model(
                input_ids=context_ids,
                past_key_values=cache,
                use_cache=True,
            )

    return outputs, outputs.past_key_values


@torch.no_grad()
def benchmark_latency_memory(
    model,
    input_ids,
    context_len: int,
    target_len: int,
    prefill_press=None,
    decoding_press=None,
    position_mode: str = "absolute",
):
    device = next(model.parameters()).device
    input_ids = input_ids[:, : context_len + target_len].to(device)

    context_ids = input_ids[:, :context_len]
    target_ids = input_ids[:, context_len : context_len + target_len]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # ===== Prefill time: context prefill + optional prefill compression =====
    start = time.perf_counter()

    _, past = prefill_context(
        model=model,
        context_ids=context_ids,
        prefill_press=prefill_press,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    prefill_time_s = time.perf_counter() - start
    prefill_stats = cache_length_stats(past, model.config.num_hidden_layers)

    if position_mode == "absolute":
        position_base = context_len
    elif position_mode == "compressed":
        position_base = past.get_seq_length(0)
    else:
        raise ValueError(f"Unknown position_mode: {position_mode}")

    # ===== TPOT: teacher-forced target loop, no CE loss =====
    decode_ctx = decoding_press(model) if decoding_press is not None else contextlib.nullcontext()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    decode_start = time.perf_counter()

    with decode_ctx:
        for i in range(target_ids.shape[1] - 1):
            position_ids = torch.tensor(
                [[position_base + i]],
                dtype=torch.long,
                device=device,
            )

            outputs = model(
                input_ids=target_ids[:, i : i + 1],
                past_key_values=past,
                position_ids=position_ids,
                use_cache=True,
            )

            past = outputs.past_key_values

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    decode_time_s = time.perf_counter() - decode_start
    decode_steps = max(1, target_ids.shape[1] - 1)

    tpot_ms = decode_time_s * 1000.0 / decode_steps
    throughput = decode_steps / decode_time_s if decode_time_s > 0 else float("inf")

    peak_mem_mb = None
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    final_stats = cache_length_stats(past, model.config.num_hidden_layers)

    return {
        # Keep old names for compatibility.
        "ttft_s": prefill_time_s,
        "prefill_time_s": prefill_time_s,
        "tpot_ms": tpot_ms,
        "throughput_tok_s": throughput,
        "peak_mem_mb": peak_mem_mb,

        # Old compatibility fields: layer0 length.
        "prefill_cache_len": prefill_stats["layer0"],
        "final_cache_len": final_stats["layer0"],

        # New robust cache statistics.
        "prefill_cache_len_layer0": prefill_stats["layer0"],
        "prefill_cache_len_avg": prefill_stats["avg"],
        "prefill_cache_len_min": prefill_stats["min"],
        "prefill_cache_len_max": prefill_stats["max"],
        "prefill_cache_lens": str(prefill_stats["all"]),

        "final_cache_len_layer0": final_stats["layer0"],
        "final_cache_len_avg": final_stats["avg"],
        "final_cache_len_min": final_stats["min"],
        "final_cache_len_max": final_stats["max"],
        "final_cache_lens": str(final_stats["all"]),
    }


def append_csv(path: str, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Loading model/tokenizer")
    print(f"model: {args.model_name}")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
    ).to(device).eval()

    total_len = args.context_len + args.target_len

    print("=" * 80)
    print("Loading dataset")
    print(f"dataset: {args.dataset}")

    text = load_text(args.dataset, args.dataset_path)
    input_ids = tokenize_window(
        tokenizer=tokenizer,
        text=text,
        total_len=total_len,
        token_offset=args.token_offset,
    ).to(device)

    print(f"token window length: {input_ids.shape[1]}")
    print(f"context_len: {args.context_len}")
    print(f"target_len:  {args.target_len}")

    print("=" * 80)
    print("Building press")
    print(f"method: {args.method}")
    print(f"budget: {args.budget}")
    print(f"n_sink: {args.n_sink}")

    prefill_press, decoding_press = build_presses(
        method=args.method,
        budget=args.budget,
        n_sink=args.n_sink,
    )

    print("=" * 80)
    print("Running PPL")

    ppl = continuation_ppl(
        model=model,
        input_ids=input_ids,
        context_len=args.context_len,
        target_len=args.target_len,
        prefill_press=prefill_press,
        decoding_press=decoding_press,
        position_mode=args.position_mode,
        count_first_target=False,
    )

    # Build fresh presses for timing, so no hidden state carries over.
    prefill_press, decoding_press = build_presses(
        method=args.method,
        budget=args.budget,
        n_sink=args.n_sink,
    )

    print("=" * 80)
    print("Running latency/memory benchmark")

    perf = benchmark_latency_memory(
        model=model,
        input_ids=input_ids,
        context_len=args.context_len,
        target_len=args.target_len,
        prefill_press=prefill_press,
        decoding_press=decoding_press,
        position_mode=args.position_mode,
    )

    row = {
        "model": args.model_name,
        "dataset": args.dataset,
        "dataset_path": args.dataset_path or DEFAULT_DATASET_PATHS.get(args.dataset, ""),
        "method": args.method,
        "budget": args.budget,
        "n_sink": args.n_sink,
        "context_len": args.context_len,
        "target_len": args.target_len,
        "token_offset": args.token_offset,
        "position_mode": args.position_mode,
        "ppl": ppl,
        **perf,
    }

    print("=" * 80)
    print("Result")
    for k, v in row.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    if args.output_csv:
        append_csv(args.output_csv, row)
        print(f"\nSaved result to: {args.output_csv}")

    print("✅ run_eval finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="/mnt/d/Labs/Efficient-LLM-Inference/models/pythia-70m",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="pg19",
        choices=["pg19", "wikitext"],
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--token_offset", type=int, default=0)

    parser.add_argument(
        "--method",
        type=str,
        default="baseline",
        choices=ALL_METHODS,
    )

    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--n_sink", type=int, default=4)

    parser.add_argument("--context_len", type=int, default=1536)
    parser.add_argument("--target_len", type=int, default=512)

    parser.add_argument(
        "--position_mode",
        type=str,
        default="absolute",
        choices=["absolute", "compressed"],
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/results.csv",
    )

    args = parser.parse_args()
    main(args)
