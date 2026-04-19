def estimate_flops(model_config, input_length, max_new_tokens):
    """粗略估算前向传播的 TFLOPs"""
    d_model = model_config.hidden_size
    num_layers = model_config.num_hidden_layers
    
    flops_prefill = 24 * num_layers * (d_model ** 2) * input_length
    flops_decode = 24 * num_layers * (d_model ** 2) * max_new_tokens
    
    total_tflops = (flops_prefill + flops_decode) / (10 ** 12)
    return total_tflops