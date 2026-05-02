import hashlib
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from pythia_kvpress.presses import KNormPress


def tensor_hash_1_to_1000(t: torch.Tensor) -> int:
    """
    Convert a tensor to a stable integer hash in [1, 1000].
    """
    x = t.detach().float().cpu()
    x = torch.round(x * 1000) / 1000
    raw = x.numpy().tobytes()
    digest = hashlib.md5(raw).hexdigest()
    return int(digest[:8], 16) % 1000 + 1


def get_layer_slot_hashes(cache, layer_idx: int):
    """
    For one layer, compute one hash per cache slot.

    Each slot hash is computed from:
      concat(keys[0, :, slot_idx, :], values[0, :, slot_idx, :])

    For KNorm, selection is per-head, so this is a slot-content hash,
    not a strict original-token-id hash.
    """
    layer = cache.layers[layer_idx]
    keys = layer.keys
    values = layer.values

    seq_len = keys.shape[2]
    hashes = []

    for slot_idx in range(seq_len):
        k = keys[0, :, slot_idx, :].reshape(-1)
        v = values[0, :, slot_idx, :].reshape(-1)
        kv = torch.cat([k, v], dim=0)
        hashes.append(tensor_hash_1_to_1000(kv))

    return hashes


def print_cache_hashes(step_name, token_text, cache, num_layers):
    print("=" * 100)
    print(f"[{step_name}] token: {repr(token_text)}")

    for layer_idx in range(num_layers):
        hashes = get_layer_slot_hashes(cache, layer_idx)
        print(f"layer {layer_idx}: {hashes}")


def save_hash_visualization(all_hashes, step_labels, save_path):
    """
    all_hashes:
      shape [num_steps, num_layers, budget]
    """
    all_hashes = np.array(all_hashes)
    num_steps, num_layers, budget = all_hashes.shape

    fig, axes = plt.subplots(
        nrows=num_layers,
        ncols=1,
        figsize=(max(12, 1.2 * num_steps), 2.2 * num_layers),
        sharex=True,
        constrained_layout=True,
    )

    if num_layers == 1:
        axes = [axes]

    for layer_idx, ax in enumerate(axes):
        mat = all_hashes[:, layer_idx, :].T  # [budget, steps]

        im = ax.imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            vmin=1,
            vmax=1000,
        )

        ax.set_title(f"Layer {layer_idx} KNorm-retained KV slot hashes")
        ax.set_ylabel("Cache slot")
        ax.set_yticks(range(budget))
        ax.set_yticklabels([str(i) for i in range(budget)])

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

    axes[-1].set_xlabel("Step")
    axes[-1].set_xticks(range(num_steps))
    axes[-1].set_xticklabels(step_labels, rotation=30, ha="right")

    cbar = fig.colorbar(im, ax=axes, shrink=0.98)
    cbar.set_label("Hash value (1-1000)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def collect_all_layer_hashes(cache, num_layers, expected_budget=None):
    all_layers = []

    for layer_idx in range(num_layers):
        hashes = get_layer_slot_hashes(cache, layer_idx)

        if expected_budget is not None and len(hashes) != expected_budget:
            print(
                f"[warning] layer={layer_idx}, "
                f"cache len={len(hashes)} != expected_budget={expected_budget}"
            )

        all_layers.append(hashes)

    return all_layers


def main():
    model_name = "/mnt/d/Labs/Efficient-LLM-Inference/models/pythia-70m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Config =====
    budget = 10
    max_new_tokens = 10
    keep_low_norm = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

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

    prompt_len = input_ids.shape[1]
    num_layers = model.config.num_hidden_layers

    print(f"Prompt token length: {prompt_len}")
    print(f"KNorm budget: {budget}")
    print(f"keep_low_norm: {keep_low_norm}")

    cache = DynamicCache(config=model.config)

    # ===== 1) Prefill with KNorm compression =====
    prefill_press = KNormPress(
        mode="prefill",
        budget=budget,
        keep_low_norm=keep_low_norm,
    )

    with torch.no_grad():
        with prefill_press(model):
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
            )

    past = outputs.past_key_values

    print(f"Prefill cache length layer 0: {past.get_seq_length(0)}")

    all_hashes = []
    step_labels = []

    prefill_hashes = collect_all_layer_hashes(
        past,
        num_layers=num_layers,
        expected_budget=budget,
    )
    all_hashes.append(prefill_hashes)
    step_labels.append("prefill")

    print_cache_hashes(
        step_name="prefill-after-KNorm",
        token_text="<prefill>",
        cache=past,
        num_layers=num_layers,
    )

    # Greedy first generated token from prefill logits.
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # ===== 2) Decode with online KNorm compression =====
    decode_press = KNormPress(
        mode="decode",
        budget=budget,
        keep_low_norm=keep_low_norm,
    )

    generated_token_texts = []

    with torch.no_grad():
        with decode_press(model):
            for step in range(max_new_tokens):
                token_id = next_token.item()
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                generated_token_texts.append(token_text)

                # No rerotation: use original absolute positions.
                position_ids = torch.tensor(
                    [[prompt_len + step]],
                    dtype=torch.long,
                    device=device,
                )

                outputs = model(
                    input_ids=next_token,
                    past_key_values=past,
                    position_ids=position_ids,
                    use_cache=True,
                )

                past = outputs.past_key_values

                layer_hashes = collect_all_layer_hashes(
                    past,
                    num_layers=num_layers,
                    expected_budget=budget,
                )
                all_hashes.append(layer_hashes)
                step_labels.append(f"d{step + 1}")

                print_cache_hashes(
                    step_name=f"decode-step-{step + 1}",
                    token_text=token_text,
                    cache=past,
                    num_layers=num_layers,
                )

                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # ===== 3) Save visualization =====
    save_path = "tests/artifacts/knorm_trace_hashes.png"
    save_hash_visualization(
        all_hashes=all_hashes,
        step_labels=step_labels,
        save_path=save_path,
    )

    print("\nGenerated token pieces:")
    for i, tok in enumerate(generated_token_texts, start=1):
        print(f"step {i}: {repr(tok)}")

    print(f"\nSaved visualization to: {save_path}")
    print("✅ KNorm trace test finished.")


if __name__ == "__main__":
    main()