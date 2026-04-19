import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

class StreamingLLMLayer(DynamicLayer):
    """
    具体的物理抽屉：负责在每次 update 时，狠狠地把中间的 Token 裁掉。
    """
    def __init__(self, n_sink: int, window_size: int):
        super().__init__()
        self.n_sink = n_sink
        self.window_size = window_size
        self.max_capacity = n_sink + window_size

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        # 1. 基础拼接：先让官方的底层代码把新的 KV 拼接到历史 KV 上
        super().update(key_states, value_states, *args, **kwargs)

        # 2. 核心切片逻辑：如果当前长度超过了预算，执行截断！
        seq_len = self.keys.shape[-2]
        if seq_len > self.max_capacity:
            # torch.cat 把开头的 n_sink 个，和结尾的 window_size 个重新拼起来
            self.keys = torch.cat([
                self.keys[..., :self.n_sink, :],
                self.keys[..., -self.window_size:, :]
            ], dim=-2)
            
            self.values = torch.cat([
                self.values[..., :self.n_sink, :],
                self.values[..., -self.window_size:, :]
            ], dim=-2)

        # 返回精简过后的 KV 交给 Attention 去算
        return self.keys, self.values


class StreamingLLMCache(DynamicCache):
    """
    外层文件柜：负责管理 6 个抽屉，并“欺骗”大模型的位置编码 (RoPE)。
    """
    def __init__(self, num_hidden_layers: int, budget: int, n_sink: int = 4):
        super().__init__()
        # 你的总预算 budget = n_sink + window_size
        # 比如预算 256，sink 留 4 个，那 window 尾巴就留 252 个
        window_size = max(0, budget - n_sink)
        
        # 霸道总裁时刻：把官方的抽屉换成我们刚刚写的截断抽屉
        self.layers = [StreamingLLMLayer(n_sink, window_size) for _ in range(num_hidden_layers)]
        
        # ⚠️ 极其关键的变量：记录真实输入过的 Token 总数
        self._seen_tokens = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        # 只有在第 0 层的时候，我们记录一次新进来的词数（防止 6 层重复加 6 次）
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs=cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # ⚠️ 核心欺骗：当大模型问“句子现在多长了？”时，
        # 绝对不能返回抽屉里真实的张量长度(比如256)，必须返回历史总长度！
        # 否则 RoPE 旋转位置编码会彻底错乱，模型瞬间变成傻子。
        return self._seen_tokens

    def get_max_cache_shape(self) -> int:
        return self.layers[0].max_capacity