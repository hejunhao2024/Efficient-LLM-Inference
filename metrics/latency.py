import time
from transformers.generation.streamers import BaseStreamer

class PerfStreamer(BaseStreamer):
    """用于精确捕捉 TTFT 和 TPOT 的探针"""
    def __init__(self):
        self.start_time = 0.0
        self.first_token_time = 0.0
        self.end_time = 0.0
        self.token_count = 0
        self.is_first_chunk = True

    def put(self, value):
        current_time = time.perf_counter()
        if self.is_first_chunk:
            self.first_token_time = current_time
            self.is_first_chunk = False
        else:
            self.token_count += value.shape[0] if len(value.shape) > 0 else 1

    def end(self):
        self.end_time = time.perf_counter()

def calculate_latency_metrics(streamer, max_new_tokens):
    """计算延迟指标并返回 (ttft_s, tpot_ms, throughput)"""
    ttft_s = streamer.first_token_time - streamer.start_time
    total_decode_time = streamer.end_time - streamer.first_token_time
    
    decode_tokens = max(1, max_new_tokens - 1) 
    tpot_ms = (total_decode_time / decode_tokens) * 1000
    
    throughput = max_new_tokens / (streamer.end_time - streamer.start_time)
    return ttft_s, tpot_ms, throughput