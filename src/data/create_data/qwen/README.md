# Qwen Infographic Data Generation

This folder contains tools for generating infographic data using Qwen models with vLLM. The pipeline includes creating summaries from Wikipedia and then generating infographic descriptions.

## Overview

The infographic data generation pipeline consists of 4 main steps:
1. **Generate Wikipedia summaries**: `generate_wikipedia_summary.py`
2. **Generate infographic descriptions**: `generate_infographic_data.py` 
3. **Extract bounding boxes**: `extract_bboxes.py`
4. **Merge data**: `merge_infographic_bboxes.py`

## Installation

Install the required dependencies:

```bash
cd /home/thinhnp/hf_vqa/src/data/create_data/qwen
pip install -r requirements.txt
```

**Note**: Requires GPU with at least 16GB VRAM to run Qwen3-8B

## Usage Guide

### Step 1: Generate Wikipedia summaries

```bash
export PYTHONPATH="./:$PYTHONPATH"

python generate_wikipedia_summary.py \
    --model_name "unsloth/Qwen3-8B" \
    --input_data "/home/thinhnp/hf_vqa/src/data/create_data/wikipedia/wikipedia_full_processed.json" \
    --template_path "/home/thinhnp/hf_vqa/src/prompts/summary.jinja" \
    --start-wiki 0 \
    --end-wiki 232000 \
    --batch_size 8 \
    --output-dir "src/data/create_data/output/summarize"
```

**Important parameters:**
- `--model_name`: Qwen model name (default: `unsloth/Qwen3-8B`)
- `--input_data`: Path to processed Wikipedia data
- `--start-wiki`: Start index (inclusive)
- `--end-wiki`: End index (exclusive)
- `--batch_size`: Batch size (reduce if running out of VRAM)
- `--output-dir`: Output directory

### Step 2: Generate infographic descriptions

```bash
export PYTHONPATH="./:$PYTHONPATH"

python generate_infographic_data.py \
    --model_name "unsloth/Qwen3-8B" \
    --input_data "src/data/create_data/output/summarize" \
    --template_path "/home/thinhnp/hf_vqa/src/prompts/bizgen.jinja" \
    --start-wiki 0 \
    --end-wiki 232000 \
    --batch_size 8 \
    --output-dir "src/data/create_data/output/infographic"
```

### Step 3: Extract bounding boxes (run only once)

```bash
python extract_bboxes.py
```

This script reads from `bizgen/meta/infographics.json` and `bizgen/meta/infographics_multilang.json` to create `extracted_bboxes.json`.

### Step 4: Merge infographic data with bounding boxes

```bash
python merge_infographic_bboxes.py \
    --start-wiki 0 \
    --end-wiki 232000 \
    --seed 42 \
    --output-dir "src/data/create_data/output/bizgen_format"
```

**Advanced parameters:**
- `--layout_data`: Include data layers (may affect image quality)
- `--layout_timeline`: Include timeline layers (may affect image quality)
- `--seed`: Random seed for reproducibility

## Complete Usage Example

```bash
export PYTHONPATH="./:$PYTHONPATH"

# Step 1: Generate Wikipedia summaries (parallel execution)
CUDA_VISIBLE_DEVICES=0 python generate_wikipedia_summary.py \
    --start-wiki 0 --end-wiki 1000 --batch_size 4

# Step 2: Generate infographic descriptions
CUDA_VISIBLE_DEVICES=0 python generate_infographic_data.py \
    --start-wiki 0 --end-wiki 1000 --batch_size 8

# Step 3: Extract bounding boxes (run only once)
python extract_bboxes.py

# Step 4: Merge data
python merge_infographic_bboxes.py \
    --start-wiki 0 --end-wiki 1000 --seed 42
```

## Processing Large Datasets

To process tens of thousands of samples, split the work into chunks:

```bash
export PYTHONPATH="./:$PYTHONPATH"

# Example
for i in {0..25}; do
    start=$((i * 1000))
    end=$(((i + 1) * 1000))
    
    echo "Processing chunk $i: $start -> $end"
    
    # Generate summaries
    CUDA_VISIBLE_DEVICES=0 python generate_wikipedia_summary.py \
        --start-wiki $start --end-wiki $end --batch_size 4
    
    # Generate infographics
    CUDA_VISIBLE_DEVICES=0 python generate_infographic_data.py \
        --start-wiki $start --end-wiki $end --batch_size 8
    
    # Merge data
    python merge_infographic_bboxes.py \
        --start-wiki $start --end-wiki $end --seed 42
done
```

## Output Data Structure

### Wikipedia Summaries (`output/summarize/`)
```json
{
  "id": "wiki_article_id",
  "title": "Article Title",
  "categories": ["category1"],
  "generated_summary": "AI-generated summary...",
  "success": true,
  "summary_id": 1
}
```

### Infographic Descriptions (`output/infographic/`)
```json
{
  "id": "wiki_article_id", 
  "title": "Article Title",
  "categories": ["category1"],
  "generated_infographic": {
    "full_image_caption": "Overall description...",
    "layers_all": [
      {
        "category": "figure|text|data|timeline",
        "caption": "Layer description..."
      }
    ]
  },
  "success": true,
  "infographic_id": 1
}
```

### Final Data (`output/bizgen_format/`)
```json
{
  "index": 1,
  "layers_all": [
    {
      "category": "base|element|text",
      "top_left": [x1, y1],
      "bottom_right": [x2, y2], 
      "caption": "Processed caption with colors and fonts..."
    }
  ],
  "full_image_caption": "Overall description..."
}
```

## Performance Optimization

### GPU Memory
- Reduce `--batch_size` if encountering OOM errors
- Adjust `--gpu_memory_utilization` (default: 0.9)
- Use smaller `--max_model_len` if needed

### Processing Speed
- Increase `--batch_size` for faster processing
- Use multiple GPUs in parallel on different chunks
- Adjust `--temperature` and `--top_p` based on quality requirements

## Generated Files

- `output/summarize/summarize*.json`: Wikipedia summaries (50 entries/file)
- `output/infographic/infographic*.json`: Infographic descriptions (50 entries/file)
- `output/bizgen_format/wiki*.json`: Final data for image generation (50 entries/file)
- `extracted_bboxes.json`: Bounding boxes from bizgen templates

## Troubleshooting

### VRAM shortage errors
```bash
# Reduce batch size
--batch_size 2

# Reduce memory utilization
--gpu_memory_utilization 0.7
```

### Model loading errors
```bash
# Try different model
--model_name "Qwen/Qwen3-8B-Instruct"

# Or use local model
--model_name "/path/to/local/model"
```

### File not found errors
- Ensure extract_bboxes.py has been run first
- Check paths to bizgen/meta/ files
- Run from the correct repository root directory