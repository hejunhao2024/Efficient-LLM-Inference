import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


METHOD_DISPLAY = {
    "baseline": "baseline",
    "streaming_prefill": "streaming_prefill",
    "streaming_online": "streaming_online",
    "streaming_decode_only": "streaming_decode_only",
    "knorm_prefill": "knorm_prefill",
    "knorm_online": "knorm_online",
    "snapkv_prefill": "snapkv_prefill",
    "pyramidkv_prefill": "pyramidkv_prefill",
    "think_prefill": "think_prefill",
}

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

TOKEN_BUDGET_METHODS = [
    "streaming_prefill",
    "streaming_online",
    "streaming_decode_only",
    "knorm_prefill",
    "knorm_online",
    "snapkv_prefill",
    "pyramidkv_prefill",
]

# 只有 PPL 图里隐藏 knorm_online
PPL_EXCLUDE_METHODS = {"knorm_online"}

DATASET_DISPLAY = {
    "pg19": "PG19",
    "wikitext": "WikiText-2",
}

METRICS = [
    ("ppl", "PPL"),
    ("prefill_time_s", "Prefill Time (s)"),
    ("tpot_ms", "TPOT (ms)"),
    ("throughput_tok_s", "Throughput (tok/s)"),
    ("peak_mem_mb", "Peak Memory (MB)"),
    ("final_cache_len_avg", "Final Cache Length Avg"),
]


def prepare_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [
        "dataset",
        "method",
        "budget",
        "context_len",
        "ppl",
        "prefill_time_s",
        "tpot_ms",
        "throughput_tok_s",
        "peak_mem_mb",
        "final_cache_len_avg",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # 若同一配置有多次运行，这里求平均
    group_cols = ["dataset", "method", "budget", "context_len"]
    value_cols = [
        "ppl",
        "prefill_time_s",
        "tpot_ms",
        "throughput_tok_s",
        "peak_mem_mb",
        "final_cache_len_avg",
    ]
    df = df.groupby(group_cols, as_index=False)[value_cols].mean()

    df["keep_ratio"] = df["budget"] / df["context_len"]
    df["compression_ratio"] = 1.0 - df["keep_ratio"]

    return df


def get_methods_for_metric(metric_col: str):
    methods = TOKEN_BUDGET_METHODS.copy()
    if metric_col == "ppl":
        methods = [m for m in methods if m not in PPL_EXCLUDE_METHODS]
    return methods


def plot_one_metric(
    df_dataset: pd.DataFrame,
    dataset_name: str,
    metric_col: str,
    metric_label: str,
    output_dir: Path,
):
    plt.figure(figsize=(8, 5))

    methods_for_metric = get_methods_for_metric(metric_col)

    token_df = df_dataset[df_dataset["method"].isin(methods_for_metric)].copy()

    for method in METHOD_ORDER:
        if method not in methods_for_metric:
            continue

        sub = token_df[token_df["method"] == method].sort_values("budget")
        if sub.empty:
            continue

        x = sub["budget"].tolist()
        y = sub[metric_col].tolist()

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            label=METHOD_DISPLAY.get(method, method),
        )

    # baseline: 水平参考线
    baseline = df_dataset[df_dataset["method"] == "baseline"]
    if not baseline.empty:
        y = baseline.iloc[0][metric_col]
        plt.axhline(
            y=y,
            linestyle="--",
            linewidth=1.5,
            label="baseline",
        )

    # think_prefill: 水平参考线
    think = df_dataset[df_dataset["method"] == "think_prefill"]
    if not think.empty:
        y = think.iloc[0][metric_col]
        plt.axhline(
            y=y,
            linestyle=":",
            linewidth=1.8,
            label="think_prefill",
        )

    budgets = sorted(token_df["budget"].dropna().astype(int).unique().tolist())
    if budgets:
        plt.xticks(budgets)

    context_len = int(df_dataset["context_len"].iloc[0])

    plt.xlabel(f"KV Budget (smaller = stronger compression, context_len={context_len})")
    plt.ylabel(metric_label)
    plt.title(f"{dataset_name}: {metric_label} vs KV Budget")

    if metric_col == "ppl":
        plt.title(f"{dataset_name}: {metric_label} vs KV Budget (knorm_online hidden)")

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    out_path = output_dir / f"{dataset_name.lower()}_{metric_col}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")


def generate_all_plots(csv_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_df(csv_path)

    for dataset in df["dataset"].drop_duplicates():
        df_dataset = df[df["dataset"] == dataset].copy()
        dataset_name = DATASET_DISPLAY.get(dataset, dataset)

        for metric_col, metric_label in METRICS:
            plot_one_metric(
                df_dataset=df_dataset,
                dataset_name=dataset_name,
                metric_col=metric_col,
                metric_label=metric_label,
                output_dir=output_dir,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="results/all_methods.csv",
        help="Input CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/plots",
        help="Directory to save plots.",
    )
    args = parser.parse_args()

    generate_all_plots(
        csv_path=args.input_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()