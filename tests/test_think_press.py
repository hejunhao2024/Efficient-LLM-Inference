import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from pythia_kvpress.presses import ThinKPress


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
    ) * 8

    input_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[:, :64].to(device)

    cache = DynamicCache(config=model.config)

    ratio = 0.5
    press = ThinKPress(
        mode="prefill",
        key_channel_compression_ratio=ratio,
        window_size=8,
    )

    with torch.no_grad():
        with press(model):
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
            )

    past = outputs.past_key_values

    print("Cache status after ThinKPress:")
    for layer_idx in range(model.config.num_hidden_layers):
        length = past.get_seq_length(layer_idx)
        keys = past.layers[layer_idx].keys
        values = past.layers[layer_idx].values

        print(f"layer {layer_idx}:")
        print(f"  cache length: {length}")
        print(f"  keys shape:   {tuple(keys.shape)}")
        print(f"  values shape: {tuple(values.shape)}")

        assert length == 64, f"ThinK should not change seq_len, got {length}"
        assert keys.shape == values.shape, "ThinK should not change KV shape"

        # Count channels that are zero for all sequence positions.
        # keys: [B, H, S, D]
        zero_channel_mask = keys.abs().sum(dim=2) == 0  # [B, H, D]
        zero_channels_per_head = zero_channel_mask.sum(dim=-1)  # [B, H]

        print(f"  zero channels per head: {zero_channels_per_head.tolist()}")

        expected = int(keys.shape[-1] * ratio)
        assert torch.all(zero_channels_per_head == expected), (
            f"Expected {expected} zeroed key channels per head, "
            f"got {zero_channels_per_head.tolist()}"
        )

    print("✅ ThinKPress channel masking test passed.")


if __name__ == "__main__":
    main()
