import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pythia_kvpress.eval import continuation_ppl
from pythia_kvpress.presses import StreamingLLMPress, KNormPress


def main():
    model_name = "/mnt/d/Labs/Efficient-LLM-Inference/models/pythia-70m"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    text = (
        "In the old library, under a dim yellow lamp, a young student opened a worn notebook "
        "and began copying strange symbols from a forgotten manuscript. The wind moved softly "
        "outside the windows, and every few minutes the wooden floor creaked as if the building "
        "itself were listening. The student paused, looked around, and continued writing, "
        "wondering whether the hidden pattern in the text would finally reveal its meaning tonight. "
    ) * 4

    input_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    context_len = 64
    target_len = 32
    budget = 32

    print(f"input length: {input_ids.shape[1]}")
    print(f"context_len={context_len}, target_len={target_len}, budget={budget}")

    baseline_ppl = continuation_ppl(
        model=model,
        input_ids=input_ids,
        context_len=context_len,
        target_len=target_len,
        prefill_press=None,
        decoding_press=None,
        position_mode="absolute",
    )

    streaming_ppl = continuation_ppl(
        model=model,
        input_ids=input_ids,
        context_len=context_len,
        target_len=target_len,
        prefill_press=StreamingLLMPress(
            mode="prefill",
            budget=budget,
            n_sink=4,
        ),
        decoding_press=None,
        position_mode="absolute",
    )

    knorm_ppl = continuation_ppl(
        model=model,
        input_ids=input_ids,
        context_len=context_len,
        target_len=target_len,
        prefill_press=KNormPress(
            mode="prefill",
            budget=budget,
            keep_low_norm=True,
        ),
        decoding_press=None,
        position_mode="absolute",
    )

    print(f"baseline ppl:  {baseline_ppl:.4f}")
    print(f"streaming ppl: {streaming_ppl:.4f}")
    print(f"knorm ppl:     {knorm_ppl:.4f}")

    for name, ppl in [
        ("baseline", baseline_ppl),
        ("streaming", streaming_ppl),
        ("knorm", knorm_ppl),
    ]:
        assert math.isfinite(ppl), f"{name} PPL is not finite: {ppl}"
        assert ppl > 0, f"{name} PPL should be positive: {ppl}"

    print("✅ Small PPL test passed.")


if __name__ == "__main__":
    main()