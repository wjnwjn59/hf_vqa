# DeepSeek-VL2 Model Setup Guide

## Overview

DeepSeek-VL2 là Vision-Language Model thế hệ thứ 2 từ DeepSeek AI, kế thừa kiến trúc từ Janus Pro nhưng được cải tiến với khả năng hiểu ảnh tốt hơn và hỗ trợ nhiều tác vụ vision-language phức tạp hơn.

**Model Information:**
- Model: `deepseek-ai/deepseek-vl2-small`
- Size: Small variant (optimal for speed/quality trade-off)
- Type: Multimodal Causal LM (Generation 2)
- Special features: Visual grounding, multi-image understanding

---

## Installation Requirements

### 1. Install DeepSeek-VL Library

DeepSeek-VL2 yêu cầu thư viện `deepseek-vl` của DeepSeek AI:

```bash
# Option 1: Install from GitHub (recommended)
pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git

# Option 2: Clone and install locally
git clone https://github.com/deepseek-ai/DeepSeek-VL.git
cd DeepSeek-VL
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

Verify that DeepSeek-VL is installed correctly:

```python
try:
    from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
    from deepseek_vl.utils.io import load_pil_images
    print("DeepSeek-VL installation: OK")
except ImportError as e:
    print(f"DeepSeek-VL installation: FAILED - {e}")
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
    deepseek-ai/deepseek-vl2-small \
    --local-dir /mnt/dataset1/pretrained_fm/deepseek-ai_deepseek-vl2-small/ \
    --local-dir-use-symlinks False
```

### Other Available Variants

DeepSeek-VL2 has multiple size variants:

```bash
# Small variant (recommended for most tasks)
deepseek-ai/deepseek-vl2-small

# Tiny variant (fastest, lower quality)
deepseek-ai/deepseek-vl2-tiny

# Base variant (better quality, slower)
deepseek-ai/deepseek-vl2
```

Choose based on your needs and update the model path in `deepseek_vl2.py` accordingly.

### Verify Download

Check that the model files are present:

```bash
ls -lh /mnt/dataset1/pretrained_fm/deepseek-ai_deepseek-vl2-small/
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
    --model deepseek_vl2 \
    --dataset chartqapro_test \
    --batch-size 1 \
    --output-dir ./outputs/vlm_results/
```

### Run on All Datasets

```bash
#!/bin/bash
# save as: run_deepseek_vl2_all.sh

MODEL_NAME="deepseek_vl2"
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

### Similarities with Janus Pro

DeepSeek-VL2 shares architectural similarities with Janus Pro:
1. **Same conversation format** with `<|User|>` and `<|Assistant|>` roles
2. **Same processing pipeline** with `prepare_inputs_embeds()`
3. **Same generation method** through `language_model.generate()`

### Key Differences from Janus Pro

| Feature | Janus Pro | DeepSeek-VL2 |
|---------|-----------|--------------|
| Image tag | `<image_placeholder>` | `<image>` for single, `<image_placeholder>` for multi |
| Library | `janus` | `deepseek_vl` |
| Processor | `VLChatProcessor` | `DeepseekVLV2Processor` |
| Model class | `MultiModalityCausalLM` | `DeepseekVLV2ForCausalLM` |
| Visual grounding | Limited | Better support with `<|ref|>` tags |
| Multi-image | Basic | Advanced with in-context learning |

### Sample Code Structure

```python
# Load processor and model
vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Single image conversation
conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\nQuestion here",
        "images": ["path/to/image.jpg"],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# Multi-image conversation (for future use)
# conversation = [
#     {
#         "role": "<|User|>",
#         "content": "<image_placeholder> and <image_placeholder> comparison",
#         "images": ["image1.jpg", "image2.jpg"],
#     },
#     {"role": "<|Assistant|>", "content": ""},
# ]

# Process inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
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

### Issue 1: ImportError - No module named 'deepseek_vl'

**Problem:**
```
ImportError: No module named 'deepseek_vl'
```

**Solution:**
```bash
pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git
```

---

### Issue 2: Model files not found

**Problem:**
```
OSError: /mnt/dataset1/pretrained_fm/deepseek-ai_deepseek-vl2-small does not exist
```

**Solution:**
- Check model path in `deepseek_vl2.py` line 21
- Verify model is downloaded to correct location
- Use absolute path: `/mnt/dataset1/pretrained_fm/deepseek-ai_deepseek-vl2-small/`

---

### Issue 3: CUDA Out of Memory

**Problem:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:**

1. DeepSeek-VL2-small is already optimized, but check GPU memory:
```python
import torch
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

2. Clear cache before running:
```python
import torch
torch.cuda.empty_cache()
```

3. Use tiny variant for lower memory usage:
   - Change model path to `deepseek-ai/deepseek-vl2-tiny`

4. Ensure no other processes are using GPU:
```bash
nvidia-smi
```

---

### Issue 4: Wrong image tag format

**Problem:**
Model not processing images correctly or giving unexpected results.

**Solution:**

For single image (current implementation):
```python
"content": "<image>\nYour question here"
```

For multiple images (future extension):
```python
"content": "<image_placeholder> and <image_placeholder> comparison"
"images": ["image1.jpg", "image2.jpg"]
```

Current implementation uses `<image>` tag for single image tasks.

---

### Issue 5: trust_remote_code Warning

**Problem:**
```
WARNING: Loading model with trust_remote_code=True
```

**Solution:**
This is expected behavior. DeepSeek-VL2 requires custom code. This is safe when loading from official DeepSeek repository.

---

### Issue 6: Batch Processing Not Working

**Problem:**
Model seems to process only one sample at a time even with `--batch-size > 1`

**Solution:**
This is expected. DeepSeek-VL2's architecture doesn't support true batch processing for multiple images. The implementation processes samples sequentially. The `force_batchify=True` parameter is for internal processor batching, not multi-sample batching.

Use `--batch-size 1` to avoid confusion:
```bash
python vlms/run_inference.py \
    --model deepseek_vl2 \
    --dataset chartqapro_test \
    --batch-size 1
```

---

### Issue 7: Slow Inference Speed

**Observations:**
- DeepSeek-VL2-small processes ~1-2 samples per second on typical GPU
- Each sample requires separate forward passes

**Optimization Tips:**
1. Ensure model is on GPU: Check that `.cuda()` is called
2. Use `torch.no_grad()` context (already implemented)
3. Set `use_cache=True` in generation (already implemented)
4. Use smaller variant (tiny) if speed is critical
5. Use larger variant (base) if quality is more important than speed

---

## Performance Notes

### Expected Inference Speed

On NVIDIA A100 (40GB):
- DeepSeek-VL2-tiny: ~2-3 samples/second, ~1-2 GB VRAM
- DeepSeek-VL2-small: ~1-2 samples/second, ~2-4 GB VRAM
- DeepSeek-VL2-base: ~0.5-1 sample/second, ~4-6 GB VRAM

On NVIDIA V100 (32GB):
- DeepSeek-VL2-small: ~1 sample/second, ~2-4 GB VRAM

### Accuracy Characteristics

DeepSeek-VL2 is optimized for:
- Visual Question Answering
- Document understanding
- Chart interpretation
- Visual grounding (object detection from text)
- Multi-image reasoning

May be less optimal for:
- Very high-resolution images (will be resized)
- Real-time applications (slower than some alternatives)

---

## Advanced Features (Future Extensions)

### Visual Grounding

DeepSeek-VL2 supports visual grounding with special tags:

```python
conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\n<|ref|>The giraffe at the back.<|/ref|>",
        "images": ["path/to/image.jpg"],
    },
    {"role": "<|Assistant|>", "content": ""},
]
```

### Multi-Image Understanding

For comparing multiple images:

```python
conversation = [
    {
        "role": "<|User|>",
        "content": (
            "<image_placeholder>First image, "
            "<image_placeholder>Second image, "
            "What are the differences?"
        ),
        "images": ["image1.jpg", "image2.jpg"],
    },
    {"role": "<|Assistant|>", "content": ""},
]
```

**Note:** Current implementation focuses on single-image VQA tasks. Multi-image support can be added in future versions.

---

## Model Configuration

### Default Settings in `deepseek_vl2.py`

```python
model_dir = "/mnt/dataset1/pretrained_fm/deepseek-ai_deepseek-vl2-small"
dtype = torch.bfloat16
device = "cuda"
max_new_tokens = config.max_new_tokens  # From dataset config
do_sample = False  # Greedy decoding
use_cache = True  # Enable KV cache
system_prompt = ""  # No system prompt (use USER_PROMPT in content)
```

### Modifying Model Path or Variant

To use a different model variant, edit line 21 in `deepseek_vl2.py`:

```python
# For tiny variant (faster, lower quality)
def get_model(model_dir: str = "/mnt/dataset1/pretrained_fm/deepseek-ai_deepseek-vl2-tiny"):

# For base variant (slower, better quality)
def get_model(model_dir: str = "/mnt/dataset1/pretrained_fm/deepseek-ai_deepseek-vl2"):
```

Remember to download the corresponding model variant.

---

## Comparison with Other Models

| Feature | DeepSeek-VL2 | Janus Pro | LLaVA-OneVision |
|---------|--------------|-----------|-----------------|
| Batch Support | No | No | Yes |
| Custom Library | Yes (deepseek_vl) | Yes (janus) | No |
| Model Class | DeepseekVLV2ForCausalLM | MultiModalityCausalLM | AutoModelForCausalLM |
| Processor | DeepseekVLV2Processor | VLChatProcessor | AutoProcessor |
| Special Format | Yes | Yes | No |
| Visual Grounding | Yes (advanced) | Limited | No |
| Multi-Image | Yes (advanced) | Basic | Limited |
| Generation | V2 (improved) | V1 | Standard |

---

## Comparison with Janus Pro

Since both models are from DeepSeek AI and share similar architectures:

### Similarities:
- Same conversation role format (`<|User|>`, `<|Assistant|>`)
- Same processing pipeline (load_pil_images → processor → prepare_inputs_embeds)
- Same generation method (through language_model.generate())
- Both require trust_remote_code=True

### When to use DeepSeek-VL2 over Janus Pro:
- ✅ Need better visual understanding
- ✅ Working with complex visual grounding tasks
- ✅ Multi-image reasoning required
- ✅ More recent model with improvements

### When to use Janus Pro over DeepSeek-VL2:
- ✅ Simpler deployment (potentially)
- ✅ Stable version tested longer
- ✅ Specific tasks where Janus Pro was validated

---

## Additional Resources

- **Official Repository:** https://github.com/deepseek-ai/DeepSeek-VL
- **Model Card:** https://huggingface.co/deepseek-ai/deepseek-vl2-small
- **Paper:** Check DeepSeek AI publications for technical details
- **Related:** Janus Pro documentation in same directory

---

## Support

If you encounter issues not covered in this guide:

1. Check the official DeepSeek-VL GitHub issues: https://github.com/deepseek-ai/DeepSeek-VL/issues
2. Verify all dependencies are up to date
3. Test with the minimal example from the official repository first
4. Check CUDA and PyTorch compatibility
5. Compare with Janus Pro setup if similar issues

---

## Quick Reference

```bash
# Install
pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git

# Download (small variant)
huggingface-cli download deepseek-ai/deepseek-vl2-small \
    --local-dir /mnt/dataset1/pretrained_fm/deepseek-ai_deepseek-vl2-small/ \
    --local-dir-use-symlinks False

# Run
python vlms/run_inference.py \
    --model deepseek_vl2 \
    --dataset chartqapro_test \
    --batch-size 1
```

---

**Last Updated:** November 2025
**Model File:** `vlms/models/deepseek_vl2.py`
**Documentation:** `vlms/models/DEEPSEEK_VL2_SETUP.md`
**Related:** `vlms/models/JANUS_PRO_SETUP.md` (similar architecture)

