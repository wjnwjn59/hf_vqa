# Pixtral Model Setup Guide

## Overview

Pixtral-12B là Vision-Language Model từ Mistral AI, được thiết kế đặc biệt cho các tác vụ multimodal với khả năng xử lý hình ảnh và text. Implementation này sử dụng vLLM để tối ưu tốc độ inference.

**Model Information:**
- Model: `mistralai/Pixtral-12B-2409`
- Size: 12B parameters
- Type: Vision-Language Model
- Backend: vLLM (optimized for speed and memory efficiency)
- Special features: Fast inference, low memory usage

---

## Installation Requirements

### Install vLLM and Dependencies

```bash
pip install torch>=2.0.0
pip install vllm>=0.5.0
pip install pillow
```

**Features:**
- ✅ Fast inference (2-3x faster than transformers)
- ✅ Better memory efficiency (~20-24GB vs ~26-28GB)
- ✅ Optimized for production workloads
- ✅ Higher throughput

**Requirements:**
- CUDA 11.8 or higher
- Linux (Ubuntu 20.04+)
- Python 3.8-3.11
- GPU with compute capability 7.0+ (V100, T4, RTX 20xx+, A100, etc.)

---

### Install Mistral Common (Optional, for advanced tokenization)

Nếu bạn muốn sử dụng mistral_common tokenizer cho advanced use cases:

```bash
pip install --upgrade mistral_common
```

**Lưu ý:** Implementation hiện tại sử dụng vLLM's built-in tokenization. Mistral_common là optional cho advanced tokenization control.

### Check Installation

```python
try:
    from vllm import LLM, SamplingParams
    print("vLLM installation: OK")
except ImportError as e:
    print(f"vLLM installation: FAILED - {e}")
```

---

## Download Model

### Using huggingface-cli (Recommended)

```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Login (may require authentication for gated models)
huggingface-cli login

# Download model
huggingface-cli download \
    mistralai/Pixtral-12B-2409 \
    --local-dir /mnt/dataset1/pretrained_fm/mistralai_Pixtral-12B-2409/ \
    --local-dir-use-symlinks False
```

### Verify Download

Check that the model files are present:

```bash
ls -lh /mnt/dataset1/pretrained_fm/mistralai_Pixtral-12B-2409/
```

Expected files:
- `config.json`
- `model.safetensors` or `pytorch_model.bin` (multiple shards)
- `tokenizer_config.json`
- `special_tokens_map.json`
- `tokenizer.json`
- `preprocessor_config.json`

---

## Usage

### Basic Usage

```bash
cd /home/thinhnp/hf_vqa/src/eval_text-rich

python vlms/run_inference.py \
    --model pixtral \
    --dataset chartqapro_test \
    --batch-size 1 \
    --output-dir ./outputs/vlm_results/
```

### Run on All Datasets

```bash
#!/bin/bash
# save as: run_pixtral_all.sh

MODEL_NAME="pixtral"
OUTPUT_DIR="./outputs/vlm_results/"

datasets=(
    "chartqapro_test"
    "chartqa_test_human"
    "chartqa_test_augmented"
    "docvqa_val"
    "infographicvqa_val"
    "textvqa_val"
    "vqav2_restval"
)

for dataset in "${datasets[@]}"; do
    echo "===== Running inference on $dataset ====="
    python vlms/run_inference.py \
        --model $MODEL_NAME \
        --dataset $dataset \
        --batch-size 1 \
        --output-dir $OUTPUT_DIR
    
    echo "Completed $dataset"
    echo ""
done

echo "All datasets completed!"
```

---

## Architecture Details

### Pixtral with vLLM

Pixtral-12B sử dụng vLLM backend để tối ưu inference:
- **Vision Encoder:** Xử lý images thành embeddings
- **Language Model:** 12B parameter language model
- **vLLM Engine:** Optimized inference engine với PagedAttention

### Key Features

1. **vLLM Optimization:**
   - PagedAttention để quản lý KV cache hiệu quả
   - Continuous batching cho throughput cao hơn
   - Optimized CUDA kernels
   - Memory-efficient inference

2. **Processing Pipeline:**
   ```python
   from vllm import LLM, SamplingParams
   
   # Initialize vLLM
   llm = LLM(
       model=model_path,
       tokenizer_mode="mistral",
       trust_remote_code=True,
       dtype="bfloat16"
   )
   
   # Create message
   messages = [
       {
           "role": "user",
           "content": [
               {"type": "text", "text": "Question here"},
               {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
           ]
       }
   ]
   
   # Generate with vLLM
   sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
   outputs = llm.chat(messages=messages, sampling_params=sampling_params)
   response = outputs[0].outputs[0].text
   ```

---

## Troubleshooting

### Issue 1: Model size too large

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solution:**

Pixtral-12B is a large model (~24GB). Solutions:

1. **Use bfloat16 (already implemented):**
   ```python
   torch_dtype=torch.bfloat16
   ```

2. **Use device_map="auto" (already implemented):**
   ```python
   device_map="auto"  # Automatically distributes layers
   ```

3. **Ensure sufficient GPU memory:**
   - Minimum: 24GB VRAM (e.g., RTX 3090, A5000, A100)
   - Recommended: 40GB+ VRAM (e.g., A100 40GB/80GB)

4. **Clear GPU cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

5. **Check GPU usage:**
   ```bash
   nvidia-smi
   ```

---

### Issue 1b: vLLM installation issues

**Problem:**
```
ERROR: Could not build wheels for vllm
```

**Solution:**

1. **Check CUDA version:**
   ```bash
   nvcc --version
   ```
   vLLM requires CUDA 11.8 or higher.

2. **Install specific version:**
   ```bash
   pip install vllm==0.5.0 --no-build-isolation
   ```

3. **If build fails, try prebuilt wheel:**
   ```bash
   pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118
   ```

4. **Check system requirements:**
   - Linux (Ubuntu 20.04+)
   - CUDA 11.8+
   - Python 3.8-3.11
   - GPU with compute capability 7.0+ (V100, T4, RTX 20xx+, A100, etc.)

---

### Issue 2: Model files not found

**Problem:**
```
OSError: /mnt/dataset1/pretrained_fm/mistralai_Pixtral-12B-2409 does not exist
```

**Solution:**
- Check model path in `pixtral.py` line 21
- Verify model is downloaded to correct location
- Check naming: `mistralai_Pixtral-12B-2409` (underscore between publisher and model)

---

### Issue 3: Slow inference speed

**Problem:**
Inference takes very long per sample.

**Solution:**

Pixtral-12B is a large model:

1. **Expected speed on different GPUs:**
   - A100 (80GB): ~1-2 seconds/sample
   - A100 (40GB): ~2-3 seconds/sample
   - RTX 3090 (24GB): ~3-5 seconds/sample

2. **Optimization tips:**
   - Ensure model is on GPU (check with `nvidia-smi`)
   - Use `torch.no_grad()` (already implemented)
   - Batch size must be 1 (no batch support)

3. **Monitor GPU utilization:**
   ```bash
   watch -n 0.5 nvidia-smi
   ```

---

### Issue 4: trust_remote_code Warning

**Problem:**
```
WARNING: Loading model with trust_remote_code=True
```

**Solution:**
This is expected. Pixtral may require custom code. Safe when loading from official Mistral AI repository.

---

### Issue 5: Mistral Common tokenizer issues

**Problem:**
Code sample uses `mistral_common` but implementation doesn't.

**Solution:**

Current implementation uses `AutoProcessor` from transformers for compatibility with eval_text-rich. This is the recommended approach.

If you want to use mistral_common tokenizer for specific use cases:

```python
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageChunk

tokenizer = MistralTokenizer.from_model("pixtral")
# ... use for advanced tokenization
```

But for eval_text-rich, stick with AutoProcessor.

---

### Issue 6: Image format errors

**Problem:**
```
ValueError: Invalid image format
```

**Solution:**

1. Ensure images are RGB:
   ```python
   image = Image.open(image_path).convert("RGB")
   ```

2. Check image file is valid:
   ```python
   from PIL import Image
   try:
       img = Image.open(path)
       print(f"Valid image: {img.size}, {img.mode}")
   except Exception as e:
       print(f"Invalid image: {e}")
   ```

---

### Issue 7: Batch processing not working

**Problem:**
Model only processes one sample at a time.

**Solution:**

This is expected. Pixtral-12B processes samples sequentially for optimal quality. Always use `--batch-size 1`:

```bash
python vlms/run_inference.py \
    --model pixtral \
    --dataset chartqapro_test \
    --batch-size 1
```

---

## Performance Notes

### Expected Inference Speed

| GPU | VRAM | Speed | Recommended |
|-----|------|-------|-------------|
| A100 80GB | 80GB | ~1-2 sec/sample | ✅ Best |
| A100 40GB | 40GB | ~2-3 sec/sample | ✅ Good |
| A6000 | 48GB | ~2-3 sec/sample | ✅ Good |
| RTX 3090 | 24GB | ~3-5 sec/sample | ⚠️ Minimum |
| RTX 4090 | 24GB | ~2-4 sec/sample | ⚠️ Minimum |
| V100 32GB | 32GB | ~3-4 sec/sample | ✅ Acceptable |

**Note:** 24GB is the minimum. Model may not fit on GPUs with less memory.

### Memory Usage

- Model size: ~24GB (bfloat16)
- Peak memory: ~26-28GB during generation
- Recommended: 32GB+ VRAM for comfortable usage

### Accuracy Characteristics

Pixtral-12B is optimized for:
- Visual Question Answering
- Document understanding
- Chart and diagram interpretation
- General multimodal reasoning
- OCR and text recognition in images

Strengths:
- High-quality visual understanding
- Strong text recognition
- Good generalization across domains

---

## Model Configuration

### Default Settings in `pixtral.py`

```python
model_dir = "/mnt/dataset1/pretrained_fm/mistralai_Pixtral-12B-2409"
dtype = torch.bfloat16
device_map = "auto"  # Automatic GPU distribution
max_new_tokens = config.max_new_tokens  # From dataset config
do_sample = False  # Greedy decoding
trust_remote_code = True
```

### Modifying Model Path

To use a different model path, edit line 21 in `pixtral.py`:

```python
def get_model(model_dir: str = "/your/custom/path/to/model"):
```

---

## Advanced Usage (Optional)

### Using Mistral Common Tokenizer

If you want to use the mistral_common tokenizer from your code sample:

```python
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageChunk
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from PIL import Image

# Initialize tokenizer
tokenizer = MistralTokenizer.from_model("pixtral")

# Load image
image = Image.open("path/to/image.jpg")

# Create chat completion request
tokenized = tokenizer.encode_chat_completion(
    ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="Describe this image"),
                    ImageChunk(image=image),
                ]
            )
        ],
        model="pixtral",
    )
)

tokens, text, images = tokenized.tokens, tokenized.text, tokenized.images
print(f"# tokens: {len(tokens)}")
print(f"# images: {len(images)}")
```

**Note:** This is for advanced use cases. The standard implementation uses AutoProcessor for compatibility.

---

## Comparison with Other Models

| Feature | Pixtral-12B | LLaVA-OneVision | Qwen2.5-VL |
|---------|-------------|-----------------|------------|
| Size | 12B | 4B | 7B |
| Batch Support | No | Yes | Yes |
| Architecture | LLaVA-style | LLaVA-style | Qwen-style |
| Custom Library | Optional (mistral_common) | No | Yes (qwen_vl_utils) |
| Memory Requirement | 24GB+ | 8GB+ | 14GB+ |
| Speed | Medium | Fast | Fast |
| Quality | High | Medium | High |
| Best For | High-quality VQA | Fast inference | Balanced |

---

## Additional Resources

- **Official Model Card:** https://huggingface.co/mistralai/Pixtral-12B-2409
- **Mistral AI:** https://mistral.ai/
- **Mistral Common Library:** https://github.com/mistralai/mistral-common
- **Transformers Documentation:** https://huggingface.co/docs/transformers

---

## Support

If you encounter issues not covered in this guide:

1. Check Pixtral model card on Hugging Face
2. Verify GPU has sufficient memory (24GB minimum)
3. Check transformers version compatibility
4. Test with a small image first
5. Monitor GPU usage with `nvidia-smi`

---

## Quick Reference

```bash
# Install vLLM
pip install torch vllm pillow

# Optional: Install mistral_common
pip install --upgrade mistral_common

# Download model
huggingface-cli download mistralai/Pixtral-12B-2409 \
    --local-dir /mnt/dataset1/pretrained_fm/mistralai_Pixtral-12B-2409/ \
    --local-dir-use-symlinks False

# Run inference
python vlms/run_inference.py \
    --model pixtral \
    --dataset chartqapro_test \
    --batch-size 1

# Check GPU usage
nvidia-smi
```

---

## Important Notes

1. **GPU Memory:** Minimum 24GB VRAM required
2. **Backend:** vLLM (2-3x faster than transformers)
3. **Batch Size:** Always use `--batch-size 1`
4. **Speed:** ~1-3 seconds per sample depending on GPU
5. **Quality:** High-quality results, good for accuracy-critical tasks
6. **Requirements:** CUDA 11.8+, Linux, Python 3.8-3.11

---

**Last Updated:** November 2025
**Model File:** `vlms/models/pixtral.py`
**Documentation:** `vlms/models/PIXTRAL_SETUP.md`

