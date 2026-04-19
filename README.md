# Efficient-LLM-Inference
## 数据集、模型下载
```bash
hf download EleutherAI/pythia-70m   --local-dir ./pythia-70m
python data/download_dataset.py
```

Press Method List:
- StreamingLLM
- SnapKVPress
- PyramidKVPress
- ThinKPress
- AdaKVPress
- KeyRerotationPress

Model List:
- Pythia-70M
- Llama-3.1-8B
- Qwen-2.5-7B

指标:
参考加速指标
TTFT: Time To First Token
TPOT: Time Per Output Token
Throughput
Total/average FLOPs: Floating Point Operations over the sequence


# TODO
- [ ] 读kvpress源码，理解压缩策略
- [ ] 读transformers源码，看模型架构
- [ ] 实现基于DynamicCache的kvpress
- [ ] 设计kvcache新的压缩策略
