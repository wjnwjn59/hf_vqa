# VQA Data Generation

This module generates Visual Question Answering (VQA) data using QwenVL model for infographic images.

## Files

- `generate_vqa_data.py`: Main script for generating VQA data
- `../inference/qwenvl_inference.py`: QwenVL inference wrapper using vLLM

## Requirements

- vLLM
- transformers
- jinja2
- PIL (Pillow)
- torch
- tqdm

## Usage

### Basic Usage

```bash
conda activate wiki
export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python src/data/vqa/generate_vqa_data.py \
    --model_name "unsloth/Qwen2-VL-7B-Instruct" \
    --images_dir "./src/data/bizgen/output/subset_0_516" \
    --template_path "./src/prompts/vqg.jinja" \
    --num_questions 5 \
    --batch_size 4
```

**Note**: By default, the output file `vqa_data.json` will be saved directly in the `--images_dir` directory (e.g., `./src/data/bizgen/output/subset_0_516/vqa_data.json`).

### Advanced Options

```bash
CUDA_VISIBLE_DEVICES=0 python src/data/vqa/generate_vqa_data.py \
    --model_name "unsloth/Qwen2-VL-7B-Instruct" \
    --images_dir "./src/data/bizgen/output/subset_0_516" \
    --template_path "./src/prompts/vqg.jinja" \
    --output_path "./src/data/vqa/vqa_data.json" \
    --num_questions 5 \
    --batch_size 4 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_tokens 2048 \
    --gpu_memory_utilization 0.9 \
    --start_id 1 \
    --end_id 60 \
    --max_samples 100
```

## Parameters

- `--model_name`: Model name or path (default: `unsloth/Qwen2-VL-7B-Instruct`)
- `--images_dir`: Directory containing images (default: `./src/data/bizgen/output/subset_0_516`)
- `--template_path`: Path to VQG jinja template (default: `./src/prompts/vqg.jinja`)
- `--output_path`: Output JSON file path (default: `./src/data/vqa/vqa_data.json`)
- `--num_questions`: Number of questions per image (default: 5)
- `--batch_size`: Batch size for inference (default: 4)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_p`: Top-p sampling parameter (default: 0.9)
- `--max_tokens`: Maximum tokens to generate (default: 2048)
- `--gpu_memory_utilization`: GPU memory utilization (default: 0.9)
- `--start_id`: Start processing from this image ID (optional)
- `--end_id`: End processing at this image ID (optional)
- `--max_samples`: Maximum number of images to process (optional)

## Output Format

The script generates a JSON file with the following structure:

```json
[
  {
    "image_id": 1,
    "image_filename": "1.png",
    "image_path": "/path/to/1.png",
    "num_questions_requested": 5,
    "num_questions_generated": 5,
    "success": true,
    "vqa_pairs": [
      {
        "question": "What is the main topic of this infographic?",
        "answer": "The main topic is about renewable energy sources."
      },
      {
        "question": "How many different energy sources are shown?",
        "answer": "There are 4 different energy sources shown."
      }
    ]
  }
]
```

## Features

- **Batch Processing**: Processes multiple images in batches for efficiency
- **Error Handling**: Graceful handling of failed images or generation errors
- **Flexible Filtering**: Process specific image ID ranges or limit total samples
- **Progress Tracking**: Real-time progress with tqdm
- **Comprehensive Logging**: Detailed statistics and error reporting
- **JSON Output**: Structured output format for easy integration

## Notes

- The script automatically filters out `_bbox.png` files and processes only the main infographic images
- Images are processed in ID order (sorted by filename)
- Failed generations are recorded with error messages for debugging
- Both main results and summary statistics are saved to separate JSON files