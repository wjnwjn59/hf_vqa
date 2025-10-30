# Complete NarratorVQA Pipeline Guide

This guide provides step-by-step instructions to run the complete pipeline from generating infographic images to creating conversation data for training.

## Overview

The pipeline consists of 4 main stages:
1. **Generate Images**: Create infographic images from Squad v2 QA data
2. **Generate VQA Data**: Create visual question-answering pairs from images  
3. **Convert to Training Format**: Transform data into conversation format
4. **Train Model**: Fine-tune the VLM model

---

## Prerequisites

### 1. Environment Setup

Create conda environments:

```bash
# Environment for image generation (Qwen3 + BizGen)
conda env create -f ./src/data/create_data/wiki.yaml
conda env create -f ./src/data/bizgen/bizgen.yaml

# Environment for VQA generation (QwenVL)  
# Uses same wiki environment

# Environment for training
conda env create -f ./src/training/Qwen2-VL-Finetune/environment.yaml
```

### 2. Download Dependencies

```bash
# Link or download BizGen checkpoints
ln -s /mnt/VLAI_data/BizGen/checkpoints ./src/data/bizgen/

# Verify Squad v2 data exists
ls /mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl
```

### 3. Make Scripts Executable

```bash
chmod +x scripts/run_narrator_pipeline.sh
chmod +x scripts/run_narrator_parallel.sh
```

---

## Stage 1: Generate Infographic Images

This stage generates infographic images from Squad v2 QA data using Qwen3 + BizGen.

### Option A: Parallel Processing (Recommended)

Generate 30,000 images using 3 GPUs in parallel:

```bash
# Run on all GPUs automatically
bash scripts/run_narrator_parallel.sh
```

**GPU Configuration:**
- GPU 0: Files 1-200 (10,000 images)
- GPU 1: Files 201-400 (10,000 images)  
- GPU 2: Files 401-600 (10,000 images)

**Total Output:** 30,000 images in 600 files

### Option B: Single GPU Processing

Process specific ranges on individual GPUs:

```bash
# GPU 0: files 1-200 (10,000 images)
bash scripts/run_narrator_pipeline.sh 0 1 201

# GPU 1: files 201-400 (10,000 images)
bash scripts/run_narrator_pipeline.sh 1 201 401

# GPU 2: files 401-600 (10,000 images)  
bash scripts/run_narrator_pipeline.sh 2 401 601
```

### Expected Output

After completion, you should have:

```
src/data/bizgen/output/squad_v2/
├── narrator000001/    # 50 images each
├── narrator000002/
├── ...
└── narrator000600/    # Total: 30,000 images
```

---

## Stage 2: Generate VQA Data (Currently Unavailable, Skip to stage 3)

Generate Visual Question-Answering pairs from the created images using QwenVL.

### Single GPU Processing

```bash
conda activate wiki
export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python src/data/vqa/generate_vqa_data.py \
    --model_name "unsloth/Qwen2-VL-7B-Instruct" \
    --images_dir "./src/data/bizgen/output/squad_v2" \
    --template_path "./src/prompts/vqg.jinja" \
    --output_path "./src/data/vqa/vqa_data_squad_v2.json" \
    --num_questions 5 \
    --batch_size 4 \
    --max_samples 30000
```

### Multi-GPU Processing (Recommended for Large Scale)

Split processing across multiple GPUs:

```bash
# GPU 0: First 10,000 images
CUDA_VISIBLE_DEVICES=0 python src/data/vqa/generate_vqa_data.py \
    --model_name "unsloth/Qwen2-VL-7B-Instruct" \
    --images_dir "./src/data/bizgen/output/squad_v2" \
    --template_path "./src/prompts/vqg.jinja" \
    --output_path "./src/data/vqa/vqa_data_squad_v2_part1.json" \
    --num_questions 5 \
    --batch_size 4 \
    --start_id 1 \
    --end_id 10000 &

# GPU 1: Second 10,000 images  
CUDA_VISIBLE_DEVICES=1 python src/data/vqa/generate_vqa_data.py \
    --model_name "unsloth/Qwen2-VL-7B-Instruct" \
    --images_dir "./src/data/bizgen/output/squad_v2" \
    --template_path "./src/prompts/vqg.jinja" \
    --output_path "./src/data/vqa/vqa_data_squad_v2_part2.json" \
    --num_questions 5 \
    --batch_size 4 \
    --start_id 10001 \
    --end_id 20000 &

# GPU 2: Third 10,000 images
CUDA_VISIBLE_DEVICES=2 python src/data/vqa/generate_vqa_data.py \
    --model_name "unsloth/Qwen2-VL-7B-Instruct" \
    --images_dir "./src/data/bizgen/output/squad_v2" \
    --template_path "./src/prompts/vqg.jinja" \
    --output_path "./src/data/vqa/vqa_data_squad_v2_part3.json" \
    --num_questions 5 \
    --batch_size 4 \
    --start_id 20001 \
    --end_id 30000 &

# Wait for all to complete
wait
```

### Merge VQA Results (if using multi-GPU)

```bash
conda activate wiki
export PYTHONPATH="./:$PYTHONPATH"

python -c "
import json

# Merge multiple VQA files
parts = []
for i in range(1, 4):
    with open(f'./src/data/vqa/vqa_data_squad_v2_part{i}.json', 'r') as f:
        parts.extend(json.load(f))

# Save merged file
with open('./src/data/vqa/vqa_data_squad_v2.json', 'w') as f:
    json.dump(parts, f, indent=2)

print(f'Merged {len(parts)} VQA entries')
"
```

### Expected Output

After completion, you should have:
- `./src/data/vqa/vqa_data_squad_v2.json`: Main VQA dataset
- `./src/data/vqa/vqa_data_summary.json`: Generation statistics

**Expected stats for 30,000 images:**
- Total VQA pairs: ~150,000 (5 questions per image)
- File size: ~500MB - 1GB

---

## Stage 3: Convert to Training Format

Convert the VQA data and images into conversation format for model training.

```bash
conda activate wiki
export PYTHONPATH="./:$PYTHONPATH"

python src/data/narrator/convert_to_training_format.py \
    --qa-file /mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl \
    --image-base-dir src/data/bizgen/output \
    --dataset-name squad_v2 \
    --dataset-type squad_v2 \
    --output-file src/data/narrator/output/conversation_squad_v2_train.json
```

### Expected Output

After completion, you should have:
- `src/data/narrator/output/conversation_squad_v2_train.json`: Training conversations

**Format example:**
```json
[
  {
    "id": "squad_v2_1",
    "image": "src/data/bizgen/output/squad_v2/narrator000001/1.png",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat information is presented in this infographic?"
      },
      {
        "from": "gpt", 
        "value": "This infographic presents information about renewable energy sources..."
      }
    ]
  }
]
```

---

## Stage 4: Model Training

Train the VLM model using the generated conversation data.

### Setup Training Environment

```bash
conda activate qwen2_train
cd src/training/Qwen2-VL-Finetune
```

### Configure Training

Edit training configuration if needed:
```bash
# Check/edit training script
cat ./scripts/finetune_narrator.sh
```

### Start Training

```bash
bash ./scripts/finetune_narrator.sh
```

---

## Complete Pipeline Script

For automated execution of the entire pipeline:

```bash
#!/bin/bash
set -e

echo "=== Complete NarratorVQA Pipeline ==="
echo "Stage 1: Generating Images..."

# Stage 1: Generate Images (30,000 images)
bash scripts/run_narrator_parallel.sh

echo "Stage 2: Generating VQA Data..."

# Stage 2: Generate VQA Data  
conda activate wiki
export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python src/data/vqa/generate_vqa_data.py \
    --model_name "unsloth/Qwen2-VL-7B-Instruct" \
    --images_dir "./src/data/bizgen/output/squad_v2" \
    --template_path "./src/prompts/vqg.jinja" \
    --output_path "./src/data/vqa/vqa_data_squad_v2.json" \
    --num_questions 5 \
    --batch_size 4 \
    --max_samples 30000

echo "Stage 3: Converting to Training Format..."

# Stage 3: Convert to Training Format
python src/data/narrator/convert_to_training_format.py \
    --qa-file /mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl \
    --image-base-dir src/data/bizgen/output \
    --dataset-name squad_v2 \
    --dataset-type squad_v2 \
    --output-file src/data/narrator/output/conversation_squad_v2_train.json

echo "Stage 4: Starting Model Training..."

# Stage 4: Train Model
conda activate qwen2_train
cd src/training/Qwen2-VL-Finetune
bash ./scripts/finetune_narrator.sh

echo "=== Pipeline Completed Successfully! ==="
```

---

## Troubleshooting

### Common Issues

1. **GPU Memory Error**
   ```bash
   # Reduce batch size or GPU memory utilization
   --batch_size 2
   --gpu_memory_utilization 0.7
   ```

2. **File Not Found Errors**
   ```bash
   # Verify file paths and permissions
   ls -la /mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl
   ls -la ./src/data/bizgen/checkpoints/
   ```

3. **Environment Issues**
   ```bash
   # Recreate environments if needed
   conda env remove -n wiki
   conda env create -f ./src/data/create_data/wiki.yaml
   ```

## Output Summary

After completing the full pipeline, you will have:

1. **Generated Images**: 30,000 infographic images in `src/data/bizgen/output/squad_v2/`
2. **VQA Dataset**: 150,000 question-answer pairs in `src/data/vqa/vqa_data_squad_v2.json`
3. **Training Data**: Conversation format in `src/data/narrator/output/conversation_squad_v2_train.json`
4. **Trained Model**: Fine-tuned VLM model checkpoints in `src/training/Qwen2-VL-Finetune/output/`

This complete dataset can now be used for:
- Visual question answering research
- Infographic understanding tasks
- Multi-modal model evaluation
- Further model fine-tuning