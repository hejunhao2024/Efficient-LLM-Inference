import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from pythia_kvpress.hooks import register_attention_hooks
from pythia_kvpress.cache_utils import *

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

    seen_layers = set()

    # def debug_hook(module, inputs, kwargs, output):
    #     layer_idx = module.layer_idx
    #     seen_layers.add(layer_idx)

    #     print("=" * 60)
    #     print(f"[hook] layer_idx = {layer_idx}")
    #     print(f"[hook] module type = {type(module)}")
    #     print(f"[hook] kwargs keys = {list(kwargs.keys())}")

    #     # GPT-NeoX/Pythia uses `layer_past` for cache.
    #     cache_obj = kwargs.get("layer_past", None)
    #     if cache_obj is None:
    #         cache_obj = kwargs.get("past_key_values", None)

    #     print(f"[hook] cache type = {type(cache_obj)}")

    #     if cache_obj is not None and hasattr(cache_obj, "layers"):
    #         cache_layer = cache_obj.layers[layer_idx]
    #         print(f"[hook] keys shape   = {tuple(cache_layer.keys.shape)}")
    #         print(f"[hook] values shape = {tuple(cache_layer.values.shape)}")
    #         print(f"[hook] cache len    = {cache_obj.get_seq_length(layer_idx)}")

    #     if isinstance(output, (tuple, list)):
    #         print(f"[hook] output length = {len(output)}")
    #         print(f"[hook] output[0] shape = {tuple(output[0].shape)}")
    #         if len(output) > 1:
    #             print(f"[hook] output[1] type = {type(output[1])}")

    #     return output
    def debug_hook(module, inputs, kwargs, output):
        cache = get_cache_from_kwargs(kwargs)
        if cache is None:
            return output

        layer_idx = module.layer_idx
        seen_layers.add(layer_idx)

        keys, values = get_layer_kv(cache, layer_idx)

        print(f"[layer {layer_idx}] kv shape = {keys.shape}")

        set_layer_kv(cache, layer_idx, keys, values)

        return output

    with torch.no_grad():
        with register_attention_hooks(model, debug_hook):
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
            )

    print("\nFinished forward.")
    print(f"Number of hooked layers: {len(seen_layers)}")
    print(f"Hooked layer indices: {sorted(seen_layers)}")
    print(f"Returned cache length layer 0: {outputs.past_key_values.get_seq_length(0)}")


if __name__ == "__main__":
    main()