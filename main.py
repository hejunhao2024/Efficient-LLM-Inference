import argparse
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_compressors.streaming_llm import StreamingLLMCache
# 🎯 导入你解耦好的评测模块
from metrics.latency import PerfStreamer, calculate_latency_metrics
from metrics.memory import reset_memory_stats, get_peak_memory_mb
from metrics.flops import estimate_flops

# 🎯 导入你的压缩算法库 (后续写好后取消注释)
# from compressors.token_eviction import StreamingLLMCache, SnapKVCache

DATASET_PATHS = {
    "pg19": "data/PG-19/Reminiscences_of_Pioneer_Days_in_St._Paul_by_Frank_Moore.txt",
    "wikitext": "data/WikiText-2/test.txt"
}

CACHE_METHODS = {
    "baseline": None, 
    "streaming": StreamingLLMCache, # 替换为: StreamingLLMCache
    "snapkv": None,    # 替换为: SnapKVCache
}

def main(args):
    print(f"\n[🚀 Benchmark] Model: {args.model_name} | Method: {args.method} | Data: {args.dataset}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 准备数据
    with open(DATASET_PATHS[args.dataset], "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 2. 准备模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto"
    )

    inputs = tokenizer(raw_text, return_tensors="pt", truncation=True, max_length=args.context_length).to(device)
    input_length = inputs["input_ids"].shape[-1]

    # 3. 准备魔改的 Cache 柜子
    past_key_values = None
    if args.method != "baseline" and CACHE_METHODS.get(args.method):
        past_key_values = CACHE_METHODS[args.method](
            num_hidden_layers=model.config.num_hidden_layers, budget=args.budget
        )

    # 4. 执行推理 (挂载探针)
    reset_memory_stats()
    streamer = PerfStreamer()
    streamer.start_time = time.perf_counter()

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            past_key_values=past_key_values,
            streamer=streamer,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 5. 结算 Metrics (一行调用，清清爽爽)
    ttft_s, tpot_ms, throughput = calculate_latency_metrics(streamer, args.max_new_tokens)
    peak_mem_mb = get_peak_memory_mb()
    tflops = estimate_flops(model.config, input_length, args.max_new_tokens)

    # 6. 打印成绩单
    print(f"\n📊 --- Metrics Report ---")
    print(f"⏱️  TTFT:       {ttft_s:.4f} s")
    print(f"⚡  TPOT:       {tpot_ms:.2f} ms/token")
    print(f"🚀  Throughput: {throughput:.2f} tokens/s")
    print(f"💾  Peak VRAM:  {peak_mem_mb:.2f} MB")
    print(f"🧮  FLOPs:      {tflops:.4f} TFLOPs\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--method", type=str, default="baseline", choices=list(CACHE_METHODS.keys()))
    parser.add_argument("--dataset", type=str, default="wikitext", choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--context_length", type=int, default=4000)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    
    main(parser.parse_args())