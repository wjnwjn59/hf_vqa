# Janus Pro Model Setup Guide

## Overview

Janus Pro là một Vision-Language Model từ DeepSeek AI với kiến trúc đặc biệt sử dụng multimodality causal language modeling. Model này có một số đặc điểm riêng biệt so với các VLM thông thường.

**Model Information:**
- Model: `deepseek-ai/Janus-Pro-1B`
- Size: 1B parameters
- Type: Multimodal Causal LM
- Special features: Unified image encoder-decoder architecture

---

## Installation Requirements

### 1. Install Janus Library

Janus Pro yêu cầu thư viện `janus` của DeepSeek AI:

```bash
# Option 1: Install from GitHub (recommended)
pip install git+https://github.com/deepseek-ai/Janus.git

# Option 2: Clone and install locally
git clone https://github.com/deepseek-ai/Janus.git
cd Janus
pip install -e .
```

### 2. Required Dependencies

Đảm bảo các dependencies sau được cài đặt:

```bash
pip install torch>=2.0.0
pip install transformers>=4.35.0
pip install pillow
pip install accelerate
```

### 3. Check Installation

Verify that Janus is installed correctly:

```python
try:
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from janus.utils.io import load_pil_images
    print("Janus installation: OK")
except ImportError as e:
    print(f"Janus installation: FAILED - {e}")
```

---

## Download Model

### Using huggingface-cli (Recommended)

```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Login if needed (model may require authentication)
huggingface-cli login

# Download model
huggingface-cli download \
    deepseek-ai/Janus-Pro-1B \
    --local-dir /mnt/dataset1/pretrained_fm/deepseek-ai_Janus-Pro-1B/ \
    --local-dir-use-symlinks False
```

### Verify Download

Check that the model files are present:

```bash
ls -lh /mnt/dataset1/pretrained_fm/deepseek-ai_Janus-Pro-1B/
```

Expected files:
- `config.json`
- `model.safetensors` or `pytorch_model.bin`
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
    --model janus_pro \
    --dataset chartqapro_test \
    --batch-size 1 \
    --output-dir ./outputs/vlm_results/
```

### Run on All Datasets

```bash
#!/bin/bash
# save as: run_janus_pro_all.sh

MODEL_NAME="janus_pro"
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

### Key Differences from Other VLMs

1. **Special Conversation Format:**
   - Uses `<|User|>` and `<|Assistant|>` role tags (not standard "user"/"assistant")
   - Requires `<image_placeholder>` in content for image position

2. **Image Processing:**
   - Uses custom `VLChatProcessor` instead of standard `AutoProcessor`
   - Requires `load_pil_images()` utility function
   - `force_batchify=True` is used for internal batching

3. **Model Architecture:**
   - Loaded as `MultiModalityCausalLM` (not standard `AutoModelForCausalLM`)
   - Requires explicit `prepare_inputs_embeds()` call to process images
   - Generation goes through `model.language_model.generate()` (not `model.generate()`)

### Sample Code Structure

```python
# Load processor and model
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Create conversation with special format
conversation = [
    {
        "role": "<|User|>",
        "content": "<image_placeholder>\nQuestion",
        "images": [pil_image],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# Process inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True
)

# Prepare embeddings (important step!)
inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

# Generate through language_model
outputs = model.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    # ... other parameters
)
```

---

## Troubleshooting

### Issue 1: ImportError - No module named 'janus'

**Problem:**
```
ImportError: No module named 'janus'
```

**Solution:**
```bash
pip install git+https://github.com/deepseek-ai/Janus.git
```

---

### Issue 2: Model files not found

**Problem:**
```
OSError: /mnt/dataset1/pretrained_fm/deepseek-ai_Janus-Pro-1B does not exist
```

**Solution:**
- Check model path in `janus_pro.py` line 21
- Verify model is downloaded to correct location
- Use absolute path: `/mnt/dataset1/pretrained_fm/deepseek-ai_Janus-Pro-1B/`

---

### Issue 3: CUDA Out of Memory

**Problem:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:**

1. Model is already using bfloat16 - check GPU memory:
```python
import torch
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

2. Clear cache before running:
```python
import torch
torch.cuda.empty_cache()
```

3. Use smaller images (if applicable)

4. Ensure no other processes are using GPU:
```bash
nvidia-smi
```

---

### Issue 4: trust_remote_code Warning

**Problem:**
```
WARNING: Loading model with trust_remote_code=True
```

**Solution:**
This is expected behavior. Janus Pro requires custom code. This is safe when loading from official DeepSeek repository.

---

### Issue 5: Batch Processing Not Working

**Problem:**
Model seems to process only one sample at a time even with `--batch-size > 1`

**Solution:**
This is expected. Janus Pro's architecture doesn't support true batch processing for multiple images. The implementation processes samples sequentially. The `force_batchify=True` parameter is for internal processor batching, not multi-sample batching.

Use `--batch-size 1` to avoid confusion:
```bash
python vlms/run_inference.py \
    --model janus_pro \
    --dataset chartqapro_test \
    --batch-size 1
```

---

### Issue 6: AttributeError - 'MultiModalityCausalLM' has no attribute 'generate'

**Problem:**
```
AttributeError: 'MultiModalityCausalLM' object has no attribute 'generate'
```

**Solution:**
Use `model.language_model.generate()` instead of `model.generate()`. This is already correctly implemented in `janus_pro.py`.

---

### Issue 7: Slow Inference Speed

**Observations:**
- Janus Pro 1B processes ~1-2 samples per second on typical GPU
- Each sample requires separate forward passes

**Optimization Tips:**
1. Ensure model is on GPU: Check that `.cuda()` is called
2. Use `torch.no_grad()` context (already implemented)
3. Set `use_cache=True` in generation (already implemented)
4. Consider using larger Janus Pro variants for better quality (trade-off with speed)

---

## Performance Notes

### Expected Inference Speed

On NVIDIA A100 (40GB):
- Janus-Pro-1B: ~1-2 samples/second
- Memory usage: ~2-3 GB VRAM

On NVIDIA V100 (32GB):
- Janus-Pro-1B: ~1 sample/second
- Memory usage: ~2-3 GB VRAM

### Accuracy Characteristics

Janus Pro is optimized for:
- Document understanding
- Chart interpretation
- Visual reasoning
- Multi-turn conversations

May be less optimal for:
- Very high-resolution images (will be resized)
- Real-time applications (slower than some alternatives)

---

## Model Configuration

### Default Settings in `janus_pro.py`

```python
model_dir = "/mnt/dataset1/pretrained_fm/deepseek-ai_Janus-Pro-1B"
dtype = torch.bfloat16
device = "cuda"
max_new_tokens = config.max_new_tokens  # From dataset config
do_sample = False  # Greedy decoding
use_cache = True  # Enable KV cache
```

### Modifying Model Path

To use a different model path, edit line 21 in `janus_pro.py`:

```python
def get_model(model_dir: str = "/your/custom/path/to/model"):
```

Or pass it programmatically if extending the code.

---

## Comparison with Other Models

| Feature | Janus Pro | LLaVA-OneVision | Llama-4-Scout |
|---------|-----------|-----------------|---------------|
| Batch Support | No | Yes | No |
| Custom Library | Yes (janus) | No | No |
| Model Class | MultiModalityCausalLM | AutoModelForCausalLM | Llama4ForConditionalGeneration |
| Processor | VLChatProcessor | AutoProcessor | AutoProcessor |
| Special Format | Yes | No | No |
| Embeddings Step | Required | Not needed | Not needed |

---

## Additional Resources

- **Official Repository:** https://github.com/deepseek-ai/Janus
- **Model Card:** https://huggingface.co/deepseek-ai/Janus-Pro-1B
- **Paper:** Check DeepSeek AI publications for technical details

---

## Support

If you encounter issues not covered in this guide:

1. Check the official Janus GitHub issues: https://github.com/deepseek-ai/Janus/issues
2. Verify all dependencies are up to date
3. Test with the minimal example from the official repository first
4. Check CUDA and PyTorch compatibility

---

## Quick Reference

```bash
# Install
pip install git+https://github.com/deepseek-ai/Janus.git

# Download
huggingface-cli download deepseek-ai/Janus-Pro-1B \
    --local-dir /mnt/dataset1/pretrained_fm/deepseek-ai_Janus-Pro-1B/ \
    --local-dir-use-symlinks False

# Run
python vlms/run_inference.py \
    --model janus_pro \
    --dataset chartqapro_test \
    --batch-size 1
```

---

**Last Updated:** November 2025
**Model File:** `vlms/models/janus_pro.py`
**Documentation:** `vlms/models/JANUS_PRO_SETUP.md`

