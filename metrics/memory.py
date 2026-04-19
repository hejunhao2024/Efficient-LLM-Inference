import torch

def reset_memory_stats():
    """在推理前清空并重置显存统计"""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_peak_memory_mb():
    """获取峰值显存 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0