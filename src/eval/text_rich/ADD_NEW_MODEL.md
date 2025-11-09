# Guide to Adding a New Model to the Text-Rich VQA Evaluation

## 📋 Overview

This document provides detailed instructions on how to add and integrate a new Vision-Language Model (VLM) into the Text-Rich VQA evaluation system. The system supports evaluation on 6 datasets: ChartQA, ChartQA-Pro, DocVQA, InfographicVQA, TextVQA, and VQAv2.

## 🗂️ Directory Structure

```
src/eval/text_rich/
├── vlms/
│   ├── models/                    # Directory containing model implementations
│   │   ├── __init__.py
│   │   ├── internvl.py            # Example: InternVL (supports batch, dynamic preprocessing)
│   │   ├── phi.py                 # Example: Phi-4 (single inference, chat template)
│   │   ├── qwenvl.py              # Example: Qwen2.5-VL (batch, vision processing)
│   │   └── your_model.py          # ← Your new model will be created here
│   ├── run_inference.py           # Main script to run inference
│   └── __init__.py
├── evaluation/                     # Directory containing evaluation scripts
│   ├── evaluator.py
│   └── metrics.py
├── text-rich_vqa_data/            # Directory containing datasets
├── config.py                      # Configuration file for datasets and parameters
└── ADD_NEW_MODEL.md              # This file
```

## 🚀 Process for Adding a New Model

### Step 1: Prepare Model Checkpoint

**Important**: All model checkpoints must be downloaded and saved at:

```
/mnt/dataset1/pretrained_fm/<publisher>_<model-name>/
```

**Example of actual paths:**
- InternVL: `/mnt/dataset1/pretrained_fm/OpenGVLab_InternVL3_5-8B/`
- Phi-4: `/mnt/dataset1/pretrained_fm/microsoft_Phi-4-multimodal-instruct/`
- Qwen2.5-VL: `/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct/`

**How to download a model from Hugging Face:**

```bash
# Install huggingface-cli if not already installed
pip install huggingface-hub

# Log in (if the model requires authentication)
huggingface-cli login

# Download the model to the standard directory
huggingface-cli download \
    <publisher>/<model-name> \
    --local-dir /mnt/dataset1/pretrained_fm/<publisher>_<model-name>/ \
    --local-dir-use-symlinks False
```

**Specific example:**
```bash
# Download Llama-3.2-Vision
huggingface-cli download \
    meta-llama/Llama-3.2-11B-Vision-Instruct \
    --local-dir /mnt/dataset1/pretrained_fm/meta-llama_Llama-3.2-11B-Vision-Instruct/ \
    --local-dir-use-symlinks False
```

---

### Step 2: Create the Model Implementation File

Create a new file in `vlms/models/` named `<model_name>.py`. The filename should be in lowercase, without accents, and can use underscores.

**Example**: `llama_vision.py`, `internvl.py`, `phi.py`

---

### Step 3: Implement the Code Template

There are 3 main templates depending on the model's characteristics:

#### **Template 1: Basic Model (Single Inference)**

For models that process one image at a time and do not support batch processing.

```python
"""<Model Name> Inference Module"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instances (singleton pattern to avoid reloading the model)
_model: Optional[Any] = None
_processor: Optional[Any] = None

# Standard system prompt for all models
USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)


def get_model(model_dir: str = "/mnt/dataset1/pretrained_fm/<publisher>_<model-name>") -> Tuple[Any, Any]:
    """
    Initialize and return the model and processor.
    Uses a singleton pattern so the model is loaded only once.
    
    Returns:
        Tuple[model, processor]: The loaded model and processor
    """
    global _model, _processor
    
    if _model is None or _processor is None:
        # Load processor/tokenizer
        _processor = AutoProcessor.from_pretrained(
            model_dir, 
            trust_remote_code=True
        )
        
        # Load model with common optimizations
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,           # Use bf16 to save memory
            device_map="auto",                    # Automatically distribute layers to GPU
            # _attn_implementation="flash_attention_2",  # Uncomment if the model supports Flash Attention
        ).eval()
    
    return _model, _processor


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """
    Run inference on the model.
    
    IMPORTANT: This function MUST have this exact signature:
    - Input: List of questions, List of image paths, config dict
    - Output: List of answers (same length as input)
    
    Args:
        questions: List of questions
        image_paths: List of paths to images (corresponding 1-to-1 with questions)
        config: DatasetConfig object containing max_new_tokens and other params
        
    Returns:
        List[str]: List of answers (stripped and first line taken)
    """
    model, processor = get_model()
    
    # Process each question/image (no batch)
    # Note: Some models can only process one sample at a time
    results = []
    for question, image_path in zip(questions, image_paths):
        # Load and convert image to RGB
        image = Image.open(image_path).convert("RGB")
        
        # Create prompt (adjust format according to the model's requirements)
        prompt = f"{USER_PROMPT}<image>\nQuestion: {question.strip()}\nAnswer:"
        
        # Process inputs
        inputs = processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        ).to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=config.max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,                      # Greedy decoding
            )
        
        # Decode response (excluding input tokens)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        response = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Get the first line and strip whitespace
        results.append(response.splitlines()[0].strip())
    
    return results
```

---

#### **Template 2: Model with Chat Template**

For models that require a special chat format (like Phi, Llama, and many instruction-tuned models).

```python
"""<Model Name> with Chat Template Inference Module"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model: Optional[Any] = None
_processor: Optional[Any] = None

USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)


def get_model(model_dir: str = "/mnt/dataset1/pretrained_fm/<publisher>_<model-name>") -> Tuple[Any, Any]:
    """Initialize and return the model and processor."""
    global _model, _processor
    
    if _model is None or _processor is None:
        _processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",  # Use Flash Attention if available
        ).eval().to(device)
    
    return _model, _processor


def create_chat_message(question: str) -> List[Dict[str, str]]:
    """
    Create a chat message in the model's format.
    This format is usually: [{"role": "system/user/assistant", "content": "..."}]
    
    Note: Chat formats differ between models:
    - Phi: separate system prompt, user content has <|image_1|>
    - Llama: may not need a system prompt, uses <image> tag
    - Qwen: nested structure with type and content
    """
    return [
        {"role": "system", "content": USER_PROMPT},
        {"role": "user", "content": f"<|image_1|>\nQuestion: {question.strip()}\nAnswer:"}
    ]


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """Run inference with chat template."""
    model, processor = get_model()
    
    # Process each sample (some models do not support batch with chat templates)
    question, image_path = questions[0], image_paths[0]
    
    # Create chat messages
    chat = create_chat_message(question)
    
    # Apply chat template
    prompt = processor.tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Some models add an endoftext token, which needs to be removed
    if prompt.endswith('<|endoftext|>'):
        prompt = prompt.rstrip('<|endoftext|>')
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        )
    
    # Decode (excluding input tokens)
    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    return [response.splitlines()[0].strip()]
```

---

#### **Template 3: Model with Batch Processing**

For models that support processing multiple images at once (more efficient).

```python
"""<Model Name> with Batch Processing Inference Module"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model: Optional[Any] = None
_processor: Optional[Any] = None

USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)


def get_model(model_dir: str = "/mnt/dataset1/pretrained_fm/<publisher>_<model-name>") -> Tuple[Any, Any]:
    """Initialize and return the model and processor."""
    global _model, _processor
    
    if _model is None or _processor is None:
        _processor = AutoProcessor.from_pretrained(
            model_dir, 
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,          # Customize according to the model
            max_pixels=1280 * 28 * 28,
        )
        # Set padding side if necessary for batch processing
        _processor.tokenizer.padding_side = "left"
        
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
        ).eval().to(device)
    
    return _model, _processor


def create_message_batch(questions: List[str], image_paths: List[str]) -> List[List[Dict[str, Any]]]:
    """
    Create batch messages for multiple samples.
    
    Note the format for each model:
    - Qwen: Nested structure with a "type" field
    - InternVL: Flat structure with an <image> tag
    """
    return [
        [
            {"role": "system", "content": [{"type": "text", "text": USER_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": f"Question: {question.strip()}\nAnswer:"}
                ]
            }
        ]
        for question, img_path in zip(questions, image_paths)
    ]


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """Run batch inference."""
    model, processor = get_model()
    
    # Create batch messages
    messages_batch = create_message_batch(questions, image_paths)
    
    # Apply chat template to the entire batch
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]
    
    # Load all images
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    # Process batch inputs
    inputs = processor(
        text=texts,
        images=images,
        padding=True,                          # Pad to make inputs the same length
        return_tensors="pt",
    ).to(device)
    
    # Generate for the entire batch
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        )
    
    # Decode and remove input tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    decoded = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    # Get the first line of each response
    return [text.splitlines()[0].strip() for text in decoded]
```

---

### Step 4: Important Points to Note

#### 4.1. Mandatory Signature of the `inference()` function

```python
def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
```

**Required:**
- Input: `questions` (list), `image_paths` (list), `config` (dict/object)
- Output: List[str] with number of elements = len(questions)
- Each output element must be `.strip()`-ed and be the first line

#### 4.2. Response Handling

**Always:**
```python
# Get the first line and remove whitespace
response.splitlines()[0].strip()
```

**Reason:**
- Some models generate multi-line responses
- There might be trailing newlines/spaces
- Ensures clean output for evaluation

#### 4.3. Memory Management

```python
# Use bfloat16 instead of float32
torch_dtype=torch.bfloat16

# Disable gradients
with torch.no_grad():
    outputs = model.generate(...)

# Automatic device map
device_map="auto"
```

#### 4.4. Model-specific Preprocessing

**Example: InternVL needs dynamic preprocessing:**
```python
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448):
    # Split the image into multiple patches based on aspect ratio
    # See internvl.py for details
    ...
```

**Example: Qwen needs vision processing:**
```python
from qwen_vl_utils import process_vision_info

image_inputs, video_inputs = process_vision_info(messages_batch)
inputs = processor(text=texts, images=image_inputs, videos=video_inputs, ...)
```

---

### Step 5: Testing the Model Implementation

Before running the main inference, test the model implementation:

```python
# Simple test script
if __name__ == "__main__":
    # Test with one sample
    test_questions = ["What color is the sky in this image?"]
    test_images = ["/path/to/test/image.jpg"]
    
    # Mock config
    class MockConfig:
        max_new_tokens = 100
    
    results = inference(test_questions, test_images, MockConfig())
    print(f"Question: {test_questions[0]}")
    print(f"Answer: {results[0]}")
```

Run the test:
```bash
cd /home/binhdt/hf_vqa/src/eval/text_rich
python -c "from vlms.models.your_model import inference; ..."
```

---

## 🏃 Running Inference

### Basic Syntax

```bash
cd /home/binhdt/hf_vqa/src/eval/text_rich

python vlms/run_inference.py \
    --model <model_filename_without_.py> \
    --dataset <dataset_name> \
    --batch-size <number> \
    --output-dir <output_directory>
```

### Available Datasets

```python
# From config.py
DATASET_CONFIGS = {
    "chartqapro_test",           # ChartQA-Pro test set
    "chartqa_test_human",         # ChartQA human test set
    "chartqa_test_augmented",     # ChartQA augmented test set
    "docvqa_val",                 # DocVQA validation set
    "infographicvqa_val",         # InfographicVQA validation set
    "textvqa_val",                # TextVQA validation set
    "vqav2_restval",              # VQAv2 restval set
}
```

### Example Inference Runs

#### Run on 1 dataset with batch size 1 (default):
```bash
python vlms/run_inference.py \
    --model llama_vision \
    --dataset chartqapro_test
```

#### Run with a larger batch size (if the model supports it):
```bash
python vlms/run_inference.py \
    --model qwenvl \
    --dataset docvqa_val \
    --batch-size 4
```

#### Specify an output directory:
```bash
python vlms/run_inference.py \
    --model internvl \
    --dataset textvqa_val \
    --output-dir ./outputs/my_experiments/
```

#### Run with FLOPs measurement:
```bash
python vlms/run_inference.py \
    --model phi \
    --dataset chartqa_test_human \
    --report-flops
```

### Shell script to run all datasets

```bash
#!/bin/bash
# File: run_all_datasets.sh

MODEL_NAME="your_model"
OUTPUT_DIR="./outputs/vlm_results/"
BATCH_SIZE=1

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
        --batch-size $BATCH_SIZE \
        --output-dir $OUTPUT_DIR
    
    echo "Completed $dataset"
    echo ""
done

echo "All datasets completed!"
```

Run the script:
```bash
chmod +x run_all_datasets.sh
./run_all_datasets.sh
```

---

## 📊 Output Format

### Result File Structure

Results are saved at: `<output_dir>/<model_name>_<dataset_name>.csv`

**Example**: `./outputs/vlm_results/llama_vision_chartqapro_test.csv`

### CSV Format

```csv
question_id,question,predict,gt,image_id
12345,"What is the title of the chart?","Sales Report 2024","Sales Report 2024",chart_001
12346,"How many categories are shown?","5","5",chart_002
12347,"What color is the highest bar?","blue","blue",chart_003
```

**Fields:**
- `question_id`: Unique ID of the question
- `question`: The original question
- `predict`: The predicted answer from the model
- `gt`: The ground truth answer
- `image_id`: The ID of the image
- `flops` (optional): FLOPs measurement if `--report-flops` is used

---

## 📈 Evaluation Metrics

Each dataset uses a different metric (defined in `config.py`):

| Dataset | Metric | Description |
|---------|--------|-------------|
| DocVQA | ANLS | Average Normalized Levenshtein Similarity |
| InfographicVQA | ANLS | Average Normalized Levenshtein Similarity |
| TextVQA | VQA Score | Soft accuracy with multiple ground truths |
| VQAv2 | VQA Score | Soft accuracy with multiple ground truths |
| ChartQA | Relaxed Accuracy | Fuzzy string matching |
| ChartQA-Pro | Relaxed Accuracy | Fuzzy string matching |

### Running Evaluation

```bash
# Evaluate a single result file
python evaluation/evaluator.py \
    --file ./outputs/vlm_results/your_model_chartqapro_test.csv \
    --use_llm

# Evaluate an entire directory
python evaluation/evaluator.py \
    --folder ./outputs/vlm_results/ \
    --use_llm
```

---

## 🔧 Troubleshooting

### Common Errors

#### 1. Model file not found
```
FileNotFoundError: Model file not found: vlms/models/your_model.py
```
**Solution**: Check if the filename is correct and placed in `vlms/models/`

#### 2. Model checkpoint not found
```
OSError: /mnt/dataset1/pretrained_fm/model_name does not exist
```
**Solution**: Check the model path in the `get_model()` function, ensure the model has been downloaded to the correct directory.

#### 3. CUDA Out of Memory
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Solution**:
- Reduce batch size: `--batch-size 1`
- Use bfloat16: `torch_dtype=torch.bfloat16`
- Add: `device_map="auto"`
- Clear cache: `torch.cuda.empty_cache()`

#### 4. Output length mismatch
```
AssertionError: len(results) != len(questions)
```
**Solution**: Ensure the `inference()` function returns the correct number of results, equal to the number of questions.

#### 5. Flash Attention not available
```
ImportError: flash_attn is not installed
```
**Solution**: 
- Install it: `pip install flash-attn --no-build-isolation`
- Or comment out the line `_attn_implementation="flash_attention_2"`

---

## 📚 Code Reference

### InternVL (Advanced: Dynamic Preprocessing + Batch)
- File: `vlms/models/internvl.py`
- Features: Dynamic image splitting, batch chat
- Suitable for: High-resolution images, document understanding

### Phi-4 (Chat Template + Single)
- File: `vlms/models/phi.py`  
- Features: Chat template, single inference
- Suitable for: Models with a strict chat format

### Qwen2.5-VL (Batch + Vision Processing)
- File: `vlms/models/qwenvl.py`
- Features: Batch processing, nested message format
- Suitable for: High-throughput inference

---

## ✅ Pre-submission Checklist

- [ ] Model checkpoint has been downloaded to `/mnt/dataset1/pretrained_fm/`
- [ ] The file `vlms/models/<model_name>.py` has been created
- [ ] The `get_model()` function has the correct path
- [ ] The `inference()` function has the correct signature
- [ ] A simple test runs without crashing
- [ ] The output format is correct (List[str], stripped)
- [ ] Ran a test on a small dataset (e.g., chartqa_test_human)
- [ ] Checked that the output CSV file is created
- [ ] The code has clear comments

---

## 🆘 Support

If you encounter problems:
1. Re-check the templates and examples in this file
2. Look at the code of similar models in `vlms/models/`
3. Check the logs when running inference
4. Test in small steps (load model → load image → inference 1 sample)

**Important Note**: The model path MUST follow the convention:
```
/mnt/dataset1/pretrained_fm/<publisher>_<model-name>/
```
