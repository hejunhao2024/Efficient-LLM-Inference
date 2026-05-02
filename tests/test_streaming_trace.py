import hashlib
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from pythia_kvpress.presses import StreamingLLMPress


def tensor_hash_1_to_1000(t: torch.Tensor) -> int:
    """
    Convert a tensor to a stable integer hash in [1, 1000].

    We round slightly for numerical stability, then hash bytes.
    """
    x = t.detach().float().cpu()
    x = torch.round(x * 1000) / 1000
    raw = x.numpy().tobytes()
    digest = hashlib.md5(raw).hexdigest()
    value = int(digest[:8], 16) % 1000 + 1
    return value


def get_layer_token_hashes(cache, layer_idx: int):
    """
    For one layer, compute one hash per cache token slot.

    Each token slot hash is computed from:
      concat(keys[0, :, token_idx, :], values[0, :, token_idx, :])
    """
    layer = cache.layers[layer_idx]
    keys = layer.keys      # [B, H, S, D]
    values = layer.values  # [B, H, S, D]

    seq_len = keys.shape[2]
    hashes = []

    for token_idx in range(seq_len):
        k = keys[0, :, token_idx, :].reshape(-1)
        v = values[0, :, token_idx, :].reshape(-1)
        kv = torch.cat([k, v], dim=0)
        h = tensor_hash_1_to_1000(kv)
        hashes.append(h)

    return hashes


def print_cache_hashes(step, generated_text_piece, cache, num_layers):
    print("=" * 100)
    print(f"[decode step {step}] generated token: {repr(generated_text_piece)}")

    for layer_idx in range(num_layers):
        hashes = get_layer_token_hashes(cache, layer_idx)
        print(f"layer {layer_idx}: {hashes}")


def save_hash_visualization(all_hashes, save_path, n_sink=4):
    """
    all_hashes:
        shape = [num_steps, num_layers, budget]
    Save a single figure with one subplot per layer.
    """
    all_hashes = np.array(all_hashes)  # [steps, layers, budget]
    num_steps, num_layers, budget = all_hashes.shape

    fig, axes = plt.subplots(
        nrows=num_layers,
        ncols=1,
        figsize=(12, 2.2 * num_layers),
        sharex=True,
        constrained_layout=True,
    )

    if num_layers == 1:
        axes = [axes]

    for layer_idx, ax in enumerate(axes):
        # transpose to [budget, steps]
        mat = all_hashes[:, layer_idx, :].T

        im = ax.imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            vmin=1,
            vmax=1000,
        )

        ax.set_title(f"Layer {layer_idx}")
        ax.set_ylabel("Cache slot")
        ax.set_yticks(range(budget))
        ax.set_yticklabels([str(i) for i in range(budget)])

        # draw boundary between sink and recent
        if 0 < n_sink < budget:
            ax.axhline(n_sink - 0.5, linewidth=2)

        # annotate each cell with hash value
        for r in range(budget):
            for c in range(num_steps):
                ax.text(
                    c,
                    r,
                    str(mat[r, c]),
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    axes[-1].set_xlabel("Decode step")
    axes[-1].set_xticks(range(num_steps))
    axes[-1].set_xticklabels([str(i + 1) for i in range(num_steps)])

    cbar = fig.colorbar(im, ax=axes, shrink=0.98)
    cbar.set_label("Hash value (1-1000)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    model_name = "/mnt/d/Labs/Efficient-LLM-Inference/models/pythia-70m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Config =====
    n_sink = 4
    recent_window = 6
    budget = n_sink + recent_window   # = 10
    max_new_tokens = 10

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    # 一个稍长一点的 prompt，确保 prefill 比 budget 大很多
    prompt = (
        "In the old library, under a dim yellow lamp, a young student opened a worn notebook "
        "and began copying strange symbols from a forgotten manuscript. The wind moved softly "
        "outside the windows, and every few minutes the wooden floor creaked as if the building "
        "itself were listening. The student paused, looked around, and continued writing, "
        "wondering whether the hidden pattern in the text would finally reveal its meaning tonight."
    )

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    print(f"Prompt token length: {input_ids.shape[1]}")

    # ===== 1) Prefill without compression =====
    cache = DynamicCache(config=model.config)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
        )

    past = outputs.past_key_values
    num_layers = model.config.num_hidden_layers

    print(f"Prefill cache length (layer 0): {past.get_seq_length(0)}")

    # greedy 起始 token：用 prefill 最后一个位置的 logits
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # ===== 2) Decode with online StreamingLLM compression =====
    press = StreamingLLMPress(
        mode="decode",
        budget=budget,
        n_sink=n_sink,
    )

    all_hashes = []   # [step][layer][slot]
    generated_token_texts = []

    with torch.no_grad():
        with press(model):
            for step in range(max_new_tokens):
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True,
                )

                past = outputs.past_key_values

                # 当前这一步生成出的 token（greedy）
                token_id = next_token.item()
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                generated_token_texts.append(token_text)

                # 压缩后的 cache 应该已经是 budget 大小
                layer_hashes_this_step = []
                for layer_idx in range(num_layers):
                    hashes = get_layer_token_hashes(past, layer_idx)
                    if len(hashes) != budget:
                        print(
                            f"[warning] step={step+1}, layer={layer_idx}, "
                            f"cache len={len(hashes)} != budget={budget}"
                        )
                    layer_hashes_this_step.append(hashes)

                all_hashes.append(layer_hashes_this_step)

                print_cache_hashes(
                    step=step + 1,
                    generated_text_piece=token_text,
                    cache=past,
                    num_layers=num_layers,
                )

                # 下一步 greedy token
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # ===== 3) Save figure =====
    save_path = "tests/artifacts/streaming_trace_hashes.png"
    save_hash_visualization(
        all_hashes=all_hashes,
        save_path=save_path,
        n_sink=n_sink,
    )

    print("\nGenerated token pieces:")
    for i, tok in enumerate(generated_token_texts, start=1):
        print(f"step {i}: {repr(tok)}")

    print(f"\nSaved visualization to: {save_path}")
    print("✅ Streaming trace test finished.")


if __name__ == "__main__":
    main()