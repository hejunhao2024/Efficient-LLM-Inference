import argparse
from pathlib import Path

import pandas as pd


METHOD_ORDER = [
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

DATASET_NAMES = {
    "pg19": "PG19",
    "wikitext": "WikiText-2",
}


def fmt_float(x, digits=2):
    if pd.isna(x):
        return "-"
    return f"{float(x):.{digits}f}"


def fmt_int_or_dash(x):
    if pd.isna(x):
        return "-"
    return str(int(round(float(x))))


def fmt_budget(method, budget):
    if method == "baseline":
        return "-"
    if method == "think_prefill":
        return "-"
    return str(int(budget))


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    df = df.copy()
    df["_method_rank"] = df["method"].map(method_rank).fillna(999)
    df["_budget_sort"] = df["budget"].fillna(0)
    df = df.sort_values(["_method_rank", "_budget_sort"])
    return df.drop(columns=["_method_rank", "_budget_sort"])


def make_table(df: pd.DataFrame) -> str:
    lines = []
    lines.append(
        "| Method | Budget | PPL | Prefill Time (s) | TPOT (ms) | "
        "Throughput (tok/s) | Peak Mem (MB) | Prefill Cache Avg | Final Cache Avg |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for _, row in df.iterrows():
        method = row["method"]
        budget = fmt_budget(method, row["budget"])

        lines.append(
            f"| {method} "
            f"| {budget} "
            f"| {fmt_float(row['ppl'])} "
            f"| {fmt_float(row['prefill_time_s'], 4)} "
            f"| {fmt_float(row['tpot_ms'])} "
            f"| {fmt_float(row['throughput_tok_s'])} "
            f"| {fmt_float(row['peak_mem_mb'])} "
            f"| {fmt_float(row['prefill_cache_len_avg'])} "
            f"| {fmt_float(row['final_cache_len_avg'])} |"
        )

    return "\n".join(lines)


def get_baseline(df: pd.DataFrame, dataset: str):
    sub = df[(df["dataset"] == dataset) & (df["method"] == "baseline")]
    if sub.empty:
        return None
    return sub.iloc[0]


def make_analysis(df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Analysis")
    lines.append("")

    for dataset in df["dataset"].drop_duplicates():
        name = DATASET_NAMES.get(dataset, dataset)
        base = get_baseline(df, dataset)
        if base is None:
            continue

        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"The baseline PPL is **{fmt_float(base['ppl'])}**, "
            f"with final average cache length **{fmt_float(base['final_cache_len_avg'])}** "
            f"and peak memory **{fmt_float(base['peak_mem_mb'])} MB**."
        )
        lines.append("")

        sub = df[df["dataset"] == dataset]

        # Mention common budget-512 comparison if available.
        budget512 = sub[sub["budget"] == 512]
        if not budget512.empty:
            lines.append("For budget 512:")
            lines.append("")
            for method in [
                "streaming_prefill",
                "streaming_online",
                "knorm_prefill",
                "snapkv_prefill",
                "pyramidkv_prefill",
            ]:
                row = budget512[budget512["method"] == method]
                if row.empty:
                    continue
                r = row.iloc[0]
                lines.append(
                    f"- `{method}`: PPL **{fmt_float(r['ppl'])}**, "
                    f"final cache avg **{fmt_float(r['final_cache_len_avg'])}**, "
                    f"peak memory **{fmt_float(r['peak_mem_mb'])} MB**."
                )
            lines.append("")

        # ThinK
        think = sub[sub["method"] == "think_prefill"]
        if not think.empty:
            r = think.iloc[0]
            lines.append(
                f"`think_prefill` keeps the cache length unchanged "
                f"(final cache avg **{fmt_float(r['final_cache_len_avg'])}**) "
                f"and gives PPL **{fmt_float(r['ppl'])}**. "
                f"This is expected because the current ThinK implementation only masks key channels "
                f"without changing the dense KV tensor shape."
            )
            lines.append("")

        # KNorm online
        ko = sub[sub["method"] == "knorm_online"]
        if not ko.empty:
            worst = ko.sort_values("ppl", ascending=False).iloc[0]
            lines.append(
                f"`knorm_online` is unstable in this experiment. "
                f"For example, at budget {int(worst['budget'])}, its PPL is "
                f"**{fmt_float(worst['ppl'])}**. "
                f"This suggests that repeatedly reselecting KV tokens by key norm during decoding "
                f"can severely hurt language modeling quality."
            )
            lines.append("")

    lines.append(
        "Overall, the most stable optimization effects are shown by cache-length reduction "
        "and peak-memory reduction. Runtime improvements are mixed because this project uses "
        "Python-level hooks and PyTorch tensor operations; on the small Pythia-70M model, "
        "the overhead of compression can offset the attention computation saved by shorter KV caches."
    )
    lines.append("")

    return "\n".join(lines)


def generate_markdown(input_csv: str) -> str:
    df = pd.read_csv(input_csv)

    required_cols = [
        "dataset",
        "method",
        "budget",
        "ppl",
        "prefill_time_s",
        "tpot_ms",
        "throughput_tok_s",
        "peak_mem_mb",
        "prefill_cache_len_avg",
        "final_cache_len_avg",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    parts = []
    parts.append("# Results")
    parts.append("")
    parts.append("This file is automatically generated from `results/all_methods.csv`.")
    parts.append("")
    parts.append("## Experimental Setup")
    parts.append("")
    parts.append("- Model: Pythia-70M")
    parts.append("- Context length: 1536")
    parts.append("- Target length: 512")
    parts.append("- Datasets: PG19 and WikiText-2")
    parts.append("- Position mode: absolute")
    parts.append("")
    parts.append("## Full Experimental Results")
    parts.append("")

    for dataset in df["dataset"].drop_duplicates():
        name = DATASET_NAMES.get(dataset, dataset)
        sub = sort_results(df[df["dataset"] == dataset])
        parts.append(f"### {name} Results")
        parts.append("")
        parts.append(make_table(sub))
        parts.append("")

    parts.append(make_analysis(df))

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/all_methods.csv",
        help="Input CSV file produced by run_eval.py or run_all_methods.sh.",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="results/result.md",
        help="Output markdown file.",
    )
    args = parser.parse_args()

    md = generate_markdown(args.input_csv)

    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")

    print(f"Generated markdown report: {output_path}")


if __name__ == "__main__":
    main()