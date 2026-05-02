import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from pythia_kvpress.presses import PyramidKVPress


def main():
    model_name = "/mnt/d/Labs/Efficient-LLM-Inference/models/pythia-70m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    text = (
        "In the old library, a student opened a notebook and copied strange symbols. "
        "The wind moved outside the windows while the wooden floor creaked softly. "
    ) * 16

    input_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[:, :128].to(device)

    cache = DynamicCache(config=model.config)

    budget = 32
    window_size = 8

    press = PyramidKVPress(
        mode="prefill",
        budget=budget,
        window_size=window_size,
        kernel_size=3,
        beta=20,
    )

    with torch.no_grad():
        with press(model):
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
            )

    past = outputs.past_key_values

    lengths = []
    print("Cache lengths after PyramidKVPress:")
    for layer_idx in range(model.config.num_hidden_layers):
        length = past.get_seq_length(layer_idx)
        lengths.append(length)
        print(f"layer {layer_idx}: {length}")

        assert length <= input_ids.shape[1], (
            f"Layer {layer_idx} length should not exceed original length."
        )
        assert length > window_size, (
            f"Layer {layer_idx} length should keep observation window plus history."
        )

    avg_len = sum(lengths) / len(lengths)

    print(f"lengths: {lengths}")
    print(f"average length: {avg_len:.2f}")
    print(f"target average budget: {budget}")
    print("layer 0 keys shape:", tuple(past.layers[0].keys.shape))
    print("last layer keys shape:", tuple(past.layers[-1].keys.shape))

    assert lengths[0] >= lengths[-1], (
        "PyramidKV should allocate more KV to lower layers than higher layers."
    )

    # For small num_layers, rounding/capping can cause some deviation.
    assert abs(avg_len - budget) <= budget * 0.35, (
        f"Average length {avg_len:.2f} is too far from target budget {budget}."
    )

    print("✅ PyramidKVPress prefill compression test passed.")


if __name__ == "__main__":
    main()
