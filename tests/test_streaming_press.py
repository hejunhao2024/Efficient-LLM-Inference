import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from pythia_kvpress.presses import StreamingLLMPress


def main():
    model_name = "/mnt/d/Labs/Efficient-LLM-Inference/models/pythia-70m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    text = "Hello world. " * 100
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[:, :64].to(device)

    cache = DynamicCache(config=model.config)

    press = StreamingLLMPress(
        mode="prefill",
        budget=16,
        n_sink=4,
    )

    with torch.no_grad():
        with press(model):
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
            )

    past = outputs.past_key_values

    print("Cache lengths after StreamingLLMPress:")
    for layer_idx in range(model.config.num_hidden_layers):
        length = past.get_seq_length(layer_idx)
        print(f"layer {layer_idx}: {length}")
        assert length == 16, f"Expected layer {layer_idx} cache length 16, got {length}"

    k0 = past.layers[0].keys
    v0 = past.layers[0].values
    print("layer 0 keys shape:", tuple(k0.shape))
    print("layer 0 values shape:", tuple(v0.shape))

    print("✅ StreamingLLMPress prefill compression test passed.")


if __name__ == "__main__":
    main()