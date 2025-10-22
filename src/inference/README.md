## Qwen3Inference (vLLM) – Quick Guide

This README shows how to use the `Qwen3Inference` class in `src/inference` to run inference with Qwen 3 via vLLM.

### 1) Requirements
- Python 3.10+
- CUDA-enabled GPU (16GB+ VRAM recommended depending on the model)
- Libraries:
  - `vllm`
  - `transformers`

Quick install:
```bash
pip install vllm transformers
```

If using GPU, ensure your CUDA/PyTorch/vLLM versions are compatible.

### 2) Source Files
- Main source: `src/inference/qwen3_inference.py`
- Main class: `Qwen3Inference`
- Using: `export PYTHONPATH="./:$PYTHONPATH"`

### 3) Initialize the model
```python
from src.inference.qwen3_inference import Qwen3Inference

# Defaults:
# model_name='unsloth/Qwen3-8B', max_model_len=32768,
# gpu_memory_utilization=0.9, dtype='auto',
# temperature=0.7, top_p=0.9, max_tokens=8192

inference = Qwen3Inference(
    model_name='unsloth/Qwen3-8B',  # or a local model path
)
```

### 4) Quick inference
- Single prompt:
```python
answer = inference.generate_single(
    prompt="Write a short description of Hanoi",
    enable_thinking=False,
)
print(answer)
```

- Batch prompts:
```python
prompts = [
    "Explain Newton's second law",
    "Summarize the book Sapiens",
]
answers = inference.generate(prompts)
for i, a in enumerate(answers):
    print(i, a)
```

### 5) Adjust sampling parameters
You can update temperature, top_p, max_tokens on the fly:
```python
inference.update_sampling_params(temperature=0.5, top_p=0.95, max_tokens=2048)
```

Or pass a custom `SamplingParams` when calling:
```python
from vllm import SamplingParams

custom = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=1024)
answer = inference.generate_single("Hi!", custom_sampling_params=custom)
```

### 6) Tips and notes
- Defaults include `stop_token_ids=[151645, 151668]` for Qwen3.

### 7) Minimal runnable example
Create `examples/minimal_infer.py`:
```python
from src.inference.qwen3_inference import Qwen3Inference

def main():
    inf = Qwen3Inference()
    print(inf.generate_single("Hello! Please answer briefly."))

if __name__ == "__main__":
    main()
```

Run:
```bash
python examples/minimal_infer.py
```


