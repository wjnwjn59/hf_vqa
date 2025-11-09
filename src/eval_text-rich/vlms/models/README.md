# VLM Models Directory

This directory contains implementations of various Vision-Language Models for the eval_text-rich evaluation system.

## Available Models

### 1. InternVL (`internvl.py`)
- **Model:** OpenGVLab InternVL3.5-8B
- **Batch Support:** Yes
- **Special Features:** Dynamic image preprocessing, multi-patch processing
- **Path:** `/mnt/dataset1/pretrained_fm/OpenGVLab_InternVL3_5-8B`

### 2. Phi-4 (`phi.py`)
- **Model:** Microsoft Phi-4 Multimodal Instruct
- **Batch Support:** No (single inference)
- **Special Features:** Chat template, Flash Attention 2
- **Path:** `/mnt/dataset1/pretrained_fm/microsoft_Phi-4-multimodal-instruct`

### 3. Qwen2.5-VL (`qwenvl.py`)
- **Model:** Qwen2.5-VL-7B-Instruct
- **Batch Support:** Yes
- **Special Features:** Vision processing with qwen_vl_utils
- **Path:** `/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct`

### 4. MiniCPM (`minicpm.py`)
- **Model:** MiniCPM-V 2.6
- **Batch Support:** Varies by implementation
- **Path:** Check file for specific path

### 5. Molmo (`molmo.py`)
- **Model:** Molmo Vision-Language Model
- **Batch Support:** Yes
- **Path:** Check file for specific path

### 6. Ovis (`ovis.py`) - UPDATED
- **Model:** AIDC-AI Ovis1.5-Gemma2-9B
- **Batch Support:** No (sequential processing)
- **Special Features:** Separate text/visual tokenizers, conversation formatter
- **Path:** `/mnt/dataset1/pretrained_fm/AIDC-AI_Ovis1.5-Gemma2-9B`
- **Status:** Need to download

### 7. VideoLLaMA (`videollama.py`)
- **Model:** VideoLLaMA Model
- **Batch Support:** Varies by implementation
- **Path:** Check file for specific path

### 8. LLaVA-OneVision (`llava_onevision.py`) - NEW
- **Model:** lmms-lab LLaVA-OneVision-1.5-4B-Instruct
- **Batch Support:** Yes
- **Special Features:** Uses qwen_vl_utils for vision processing
- **Path:** `/mnt/dataset1/pretrained_fm/lmms-lab_LLaVA-OneVision-1.5-4B-Instruct`
- **Status:** Model already downloaded

### 9. Llama-4-Scout (`llama4_scout.py`) - NEW
- **Model:** unsloth Llama-4-Scout-17B-16E-Instruct
- **Batch Support:** No (processes samples sequentially)
- **Special Features:** Flex attention, Llama4ForConditionalGeneration
- **Path:** `/mnt/dataset1/pretrained_fm/unsloth_Llama-4-Scout-17B-16E-Instruct`
- **Status:** Need to download

### 10. Janus Pro (`janus_pro.py`) - NEW
- **Model:** deepseek-ai Janus-Pro-1B
- **Batch Support:** No (sequential processing)
- **Special Features:** Custom conversation format, requires janus library
- **Path:** `/mnt/dataset1/pretrained_fm/deepseek-ai_Janus-Pro-1B`
- **Status:** Need to download
- **Documentation:** See [JANUS_PRO_SETUP.md](./JANUS_PRO_SETUP.md)

### 11. DeepSeek-VL2 (`deepseek_vl2.py`) - NEW
- **Model:** deepseek-ai deepseek-vl2-small
- **Batch Support:** No (sequential processing)
- **Special Features:** Visual grounding, multi-image understanding, requires deepseek_vl library
- **Path:** `/mnt/dataset1/pretrained_fm/deepseek-ai_deepseek-vl2-small`
- **Status:** Need to download
- **Documentation:** See [DEEPSEEK_VL2_SETUP.md](./DEEPSEEK_VL2_SETUP.md)

### 12. Pixtral (`pixtral.py`) - NEW
- **Model:** mistralai Pixtral-12B-2409
- **Batch Support:** No (sequential processing)
- **Special Features:** vLLM optimized, fast inference, high-quality VQA
- **Path:** `/mnt/dataset1/pretrained_fm/mistralai_Pixtral-12B-2409`
- **Status:** Need to download
- **Documentation:** See [PIXTRAL_SETUP.md](./PIXTRAL_SETUP.md)
- **Hardware:** Requires 24GB+ VRAM
- **Backend:** vLLM (2-3x faster than standard transformers)

### 13. ChartGemma (`chartgemma.py`) - NEW
- **Model:** ahmed-masry chartgemma (PaliGemma fine-tuned)
- **Batch Support:** No (sequential processing)
- **Special Features:** Optimized for chart understanding, beam search decoding
- **Path:** `/mnt/dataset1/pretrained_fm/ahmed-masry_chartgemma`
- **Status:** Need to download
- **Best For:** ChartQA, chart analysis tasks

### 14. ChartInstruct-LLama2 (`chartinstruct.py`) - NEW
- **Model:** ahmed-masry ChartInstruct-LLama2 (LLaVA-based)
- **Batch Support:** No (sequential processing)
- **Special Features:** LLaVA architecture for charts, beam search, FP16 pixel values
- **Path:** `/mnt/dataset1/pretrained_fm/ahmed-masry_ChartInstruct-LLama2`
- **Status:** Need to download
- **Best For:** ChartQA, chart reasoning tasks

---

## Model Implementation Standards

All models must follow these requirements:

### 1. Function Signature
```python
def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """
    Args:
        questions: List of question strings
        image_paths: List of image file paths (1-to-1 with questions)
        config: Dataset configuration object with max_new_tokens, etc.
    
    Returns:
        List[str]: List of answer strings (same length as input)
    """
```

### 2. Singleton Pattern
```python
_model: Optional[Any] = None
_processor: Optional[Any] = None

def get_model(model_dir: str) -> Tuple[Any, Any]:
    global _model, _processor
    if _model is None or _processor is None:
        # Load model only once
    return _model, _processor
```

### 3. Standard Prompt
```python
USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, "
    "without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)
```

### 4. Output Format
- Always return first line only: `response.splitlines()[0].strip()`
- Strip all whitespace
- Return list of same length as input

---

## Usage

### Running Inference

```bash
cd /home/thinhnp/hf_vqa/src/eval_text-rich

# Basic usage
python vlms/run_inference.py \
    --model <model_name> \
    --dataset <dataset_name> \
    --batch-size <batch_size> \
    --output-dir ./outputs/vlm_results/
```

### Model Names
Use the filename without `.py` extension:
- `internvl`
- `phi`
- `qwenvl`
- `llava_onevision`
- `llama4_scout`
- `janus_pro`
- etc.

### Available Datasets
- `chartqapro_test`
- `chartqa_test_human`
- `chartqa_test_augmented`
- `docvqa_val`
- `infographicvqa_val`
- `textvqa_val`
- `vqav2_restval`

---

## Adding New Models

See [ADD_NEW_MODEL.md](../../ADD_NEW_MODEL.md) in the parent directory for detailed instructions.

**Quick Steps:**
1. Create `your_model.py` in this directory
2. Implement `get_model()` and `inference()` functions
3. Follow the standard signature and output format
4. Test with a small dataset first

---

## Model Comparison

| Model | Size | Batch | Speed | Best For |
|-------|------|-------|-------|----------|
| InternVL | 8B | Yes | Medium | High-res documents |
| Phi-4 | ~14B | No | Medium | General VQA |
| Qwen2.5-VL | 7B | Yes | Fast | Batch processing |
| LLaVA-OneVision | 4B | Yes | Fast | General VQA |
| Llama-4-Scout | 17B | No | Slow | High quality |
| Janus Pro | 1B | No | Fast | Lightweight tasks |
| DeepSeek-VL2 | Small | No | Medium | Visual grounding, multi-modal |
| Pixtral | 12B | No | Medium | High-quality VQA, OCR |
| Ovis1.5-Gemma2 | 9B | No | Medium | Multimodal understanding |
| ChartGemma | ~3B | No | Fast | Chart understanding, ChartQA |
| ChartInstruct | ~7B | No | Medium | Chart reasoning, ChartQA |

---

## Special Requirements

### Models with Custom Libraries

**Janus Pro:**
- Requires: `pip install git+https://github.com/deepseek-ai/Janus.git`
- See: [JANUS_PRO_SETUP.md](./JANUS_PRO_SETUP.md)

**DeepSeek-VL2:**
- Requires: `pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git`
- See: [DEEPSEEK_VL2_SETUP.md](./DEEPSEEK_VL2_SETUP.md)

**Pixtral:**
- Requires: `pip install vllm>=0.5.0`
- Optional: `pip install --upgrade mistral_common` (for advanced tokenization)
- See: [PIXTRAL_SETUP.md](./PIXTRAL_SETUP.md)
- **Important:** Requires 24GB+ VRAM, CUDA 11.8+

**Qwen2.5-VL & LLaVA-OneVision:**
- Requires: `pip install qwen-vl-utils`

### Models with Special Attention

**Phi-4:**
- Uses Flash Attention 2
- Install: `pip install flash-attn --no-build-isolation`

**Llama-4-Scout:**
- Uses Flex Attention
- Built into transformers library

---

## Troubleshooting

### Common Issues

1. **Model not found error:**
   - Check model path in the implementation file
   - Verify model is downloaded to `/mnt/dataset1/pretrained_fm/`

2. **CUDA out of memory:**
   - Reduce batch size to 1
   - Use models with bfloat16 dtype
   - Clear GPU cache: `torch.cuda.empty_cache()`

3. **Import errors:**
   - Install missing dependencies
   - Check Python environment

4. **Wrong output format:**
   - Verify implementation returns `List[str]`
   - Check output is stripped and first line only

---

## Testing Models

Test a model before full evaluation:

```bash
# Test with small batch
python vlms/run_inference.py \
    --model <model_name> \
    --dataset chartqapro_test \
    --batch-size 1 \
    --output-dir ./test_outputs/

# Check output file
cat ./test_outputs/<model_name>_chartqapro_test.csv | head -20
```

---

## Performance Tips

1. **Use batch processing when available:**
   - Models with batch support: InternVL, Qwen2.5-VL, LLaVA-OneVision
   - Can use `--batch-size 2` or `--batch-size 4`

2. **Optimize memory:**
   - All models use `torch.bfloat16` by default
   - Use `device_map="auto"` for automatic GPU distribution

3. **Monitor GPU usage:**
   ```bash
   nvidia-smi -l 1  # Update every second
   ```

---

## File Structure

```
vlms/models/
├── __init__.py                 # Module initialization
├── README.md                   # This file
├── JANUS_PRO_SETUP.md         # Janus Pro specific docs
├── DEEPSEEK_VL2_SETUP.md      # DeepSeek-VL2 specific docs
├── PIXTRAL_SETUP.md           # Pixtral specific docs
├── internvl.py                # InternVL implementation
├── phi.py                     # Phi-4 implementation
├── qwenvl.py                  # Qwen2.5-VL implementation
├── minicpm.py                 # MiniCPM implementation
├── molmo.py                   # Molmo implementation
├── ovis.py                    # Ovis implementation
├── videollama.py              # VideoLLaMA implementation
├── llava_onevision.py         # LLaVA-OneVision implementation (NEW)
├── llama4_scout.py            # Llama-4-Scout implementation (NEW)
├── janus_pro.py               # Janus Pro implementation (NEW)
├── deepseek_vl2.py            # DeepSeek-VL2 implementation (NEW)
├── pixtral.py                 # Pixtral implementation - vLLM optimized (NEW)
├── ovis.py                    # Ovis1.5-Gemma2 implementation (UPDATED)
├── chartgemma.py              # ChartGemma implementation (NEW)
└── chartinstruct.py           # ChartInstruct-LLama2 implementation (NEW)
```

---

## Contributing

When adding a new model:
1. Follow the implementation standards above
2. Add entry to this README
3. Create specific documentation if model has special requirements
4. Test on at least one dataset before committing

---

**Last Updated:** November 2025
**Maintained by:** VQA Evaluation Team

