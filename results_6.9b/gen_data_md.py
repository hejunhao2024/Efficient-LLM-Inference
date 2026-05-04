import argparse
import ast
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


def parse_float_list(x):
    """
    解析新格式中的 window_ppls:
    "[7.924786, 8.53902, 7.2594, 8.843704, 8.343668]"
    """
    if pd.isna(x):
        return []

    if isinstance(x, list):
        values = x
    else:
        try:
            values = ast.literal_eval(str(x))
        except Exception:
            return []

    return [float(v) for v in values]


def mean_float_list(x):
    values = parse_float_list(x)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def prepare_df(input_csv: str) -> pd.DataFrame:
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

    # 新格式：一行里包含 5 次 PPL，优先用 window_ppls 求平均
    if "window_ppls" in df.columns:
        df["ppl_raw"] = df["ppl"]
        df["ppl"] = df["window_ppls"].apply(mean_float_list)
    else:
        df["ppl"] = pd.to_numeric(df["ppl"], errors="coerce")

    numeric_cols = [
        "budget",
        "ppl",
        "prefill_time_s",
        "tpot_ms",
        "throughput_tok_s",
        "peak_mem_mb",
        "prefill_cache_len_avg",
        "final_cache_len_avg",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 如果同一 dataset/method/budget 有多行，继续求平均
    group_cols = [
        "dataset",
        "method",
        "budget",
    ]

    value_cols = [
        "ppl",
        "prefill_time_s",
        "tpot_ms",
        "throughput_tok_s",
        "peak_mem_mb",
        "prefill_cache_len_avg",
        "final_cache_len_avg",
    ]

    # 可选保留的实验设置信息
    optional_setting_cols = [
        "model",
        "context_len",
        "target_len",
        "position_mode",
        "num_eval_windows",
    ]

    existing_setting_cols = [
        c for c in optional_setting_cols
        if c in df.columns and c not in group_cols
    ]

    agg_dict = {c: "mean" for c in value_cols}
    for c in existing_setting_cols:
        agg_dict[c] = "first"

    df = df.groupby(group_cols, as_index=False).agg(agg_dict)

    return df


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
            f"The baseline mean PPL is **{fmt_float(base['ppl'])}**, "
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
                    f"- `{method}`: mean PPL **{fmt_float(r['ppl'])}**, "
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
                f"and gives mean PPL **{fmt_float(r['ppl'])}**. "
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
                f"For example, at budget {int(worst['budget'])}, its mean PPL is "
                f"**{fmt_float(worst['ppl'])}**. "
                f"This suggests that repeatedly reselecting KV tokens by key norm during decoding "
                f"can severely hurt language modeling quality."
            )
            lines.append("")

    lines.append(
        "Overall, the most stable optimization effects are shown by cache-length reduction "
        "and peak-memory reduction. Runtime improvements are mixed because this project uses "
        "Python-level hooks and PyTorch tensor operations; the overhead of compression can offset "
        "the attention computation saved by shorter KV caches."
    )
    lines.append("")

    return "\n".join(lines)


def make_setup_section(df: pd.DataFrame) -> str:
    lines = []
    lines.append("## Experimental Setup")
    lines.append("")

    if "model" in df.columns:
        models = df["model"].dropna().drop_duplicates().tolist()
        if models:
            model_names = [Path(str(m)).name for m in models]
            lines.append(f"- Model: {', '.join(model_names)}")

    if "context_len" in df.columns:
        context_lens = df["context_len"].dropna().drop_duplicates().tolist()
        if context_lens:
            context_lens = [str(int(x)) for x in context_lens]
            lines.append(f"- Context length: {', '.join(context_lens)}")

    if "target_len" in df.columns:
        target_lens = df["target_len"].dropna().drop_duplicates().tolist()
        if target_lens:
            target_lens = [str(int(x)) for x in target_lens]
            lines.append(f"- Target length: {', '.join(target_lens)}")

    datasets = [
        DATASET_NAMES.get(d, d)
        for d in df["dataset"].dropna().drop_duplicates().tolist()
    ]
    if datasets:
        lines.append(f"- Datasets: {', '.join(datasets)}")

    if "position_mode" in df.columns:
        modes = df["position_mode"].dropna().drop_duplicates().tolist()
        if modes:
            lines.append(f"- Position mode: {', '.join(map(str, modes))}")

    if "num_eval_windows" in df.columns:
        nums = df["num_eval_windows"].dropna().drop_duplicates().tolist()
        if nums:
            nums = [str(int(x)) for x in nums]
            lines.append(f"- Evaluation windows per row: {', '.join(nums)}")

    lines.append("- Reported PPL: mean of `window_ppls`")
    lines.append("")

    return "\n".join(lines)


def generate_markdown(input_csv: str) -> str:
    df = prepare_df(input_csv)

    parts = []
    parts.append("# Results")
    parts.append("")
    parts.append("This file is automatically generated from `results/all_methods.csv`.")
    parts.append("")
    parts.append(make_setup_section(df))
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