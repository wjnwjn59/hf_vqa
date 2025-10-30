# NARRATORVQA-Dataset

## Squad 2 Dataset

After downloading and extracting the [Squad 2 dataset](https://rajpurkar.github.io/SQuAD-explorer/), you will find three files in the extracted folder: `train-v2.0.json`, `dev-v2.0.json`.

```bash
/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl
/mnt/VLAI_data/Squad_v2/squad_v2_val.jsonl

/mnt/VLAI_data/ChemLit-QA
/mnt/VLAI_data/MESAQA/

/mnt/VLAI_data/AdversarialQA
/mnt/VLAI_data/NarrativeQA
```

## Using Qwen to extract layout

```python
conda env create -f ./src/data/create_data/bizgen/wiki.yaml
conda activate wiki

export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python src/data/narrator/generate_infographic_data.py \
    --model_name "unsloth/Qwen3-8B" \
    --input_data "/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl" \
    --template_path "./src/prompts/bizgen_context_qa_full.jinja" \
    --dataset_type "squad_v2" \
    --start 0 \
    --end 8 \
    --batch_size 8 
```

## Merge the bounding boxes with the generated infographic data

```bash
python src/data/narrator/merge_narrator_bboxes.py \
  --infographic-dir src/data/create_data/output/infographic \
  --start 0 \
  --end 8 \
  --seed 42
```

## Create bizgen data

```bash
conda activate bizgen

cd ./src/data/bizgen/

python inference.py \
    --ckpt_dir checkpoints/lora/infographic \
    --wiki_dir ../create_data/output/narrator_format/ \
    --subset 0:516 \
    --device cuda:2 \
    --dataset_name squad_v2
```

## Running Full Pipeline on Multiple GPUs

### Quick Start - Parallel Processing

To run the complete pipeline on 3 GPUs automatically:

```bash
# Make scripts executable
chmod +x scripts/run_narrator_pipeline.sh
chmod +x scripts/run_narrator_parallel.sh

# Run on all GPUs in parallel
bash scripts/run_narrator_parallel.sh
```

This will:
- **GPU 0**: Process images 0-9,999 (10,000 images, 200 files)
- **GPU 1**: Process images 10,000-19,999 (10,000 images, 200 files)  
- **GPU 2**: Process images 20,000-29,999 (10,000 images, 200 files)

**Total**: 30,000 images, 600 files

### Single GPU Usage

To run on a single GPU with custom range:

```bash
# Syntax: bash scripts/run_narrator_pipeline.sh <GPU_ID> <START_IDX> <END_IDX>

# GPU 0: process 10,000 images (indices 0-9999)
bash scripts/run_narrator_pipeline.sh 0 0 10000

# GPU 1: process next 10,000 images (indices 10000-19999)
bash scripts/run_narrator_pipeline.sh 1 10000 20000

# GPU 2: process final 10,000 images (indices 20000-29999)
bash scripts/run_narrator_pipeline.sh 2 20000 30000
```

### Pipeline Steps

Each GPU runs 3 steps sequentially:

1. **Generate Infographic Data** (`generate_infographic_data.py`)
   - Uses Qwen3-8B to generate infographic descriptions
   - Input: Squad v2 dataset
   - Output: `infographic*.json` files (50 entries per file)

2. **Merge Bounding Boxes** (`merge_narrator_bboxes.py`)
   - Merges generated descriptions with bbox data
   - Output: `wiki*.json` files in narrator format

3. **Generate Images** (BizGen `inference.py`)
   - Creates actual infographic images
   - Output: PNG images in `output/subset_X_Y/` folders

### Monitoring Progress

```bash
# Watch all GPUs
tail -f logs/narrator_pipeline/*_<timestamp>.log

# Watch specific GPU
tail -f logs/narrator_pipeline/gpu0_*.log
tail -f logs/narrator_pipeline/gpu1_*.log
tail -f logs/narrator_pipeline/gpu2_*.log

# Monitor GPU usage
nvidia-smi -l 1
# or
nvtop
```

### Configuration

Edit `scripts/run_narrator_parallel.sh` to customize GPU ranges:

```bash
GPU_CONFIGS=(
    "0 0 10000"       # GPU 0: images 0-9999
    "1 10000 20000"   # GPU 1: images 10000-19999
    "2 20000 30000"   # GPU 2: images 20000-29999
)
```

### Output Structure

```
src/data/create_data/output/
├── infographic/
│   ├── infographic000001.json    # 50 entries: IDs 1-50
│   ├── infographic000002.json    # 50 entries: IDs 51-100
│   └── ...
│   └── infographic000600.json    # 50 entries: IDs 29951-30000
├── narrator_format/
│   ├── wiki000001.json           # Merged format
│   ├── wiki000002.json
│   └── ...
│   └── wiki000600.json

src/data/bizgen/output/
└── squad_v2/                    # Dataset output folder
    ├── narrator000001/          # 50 images per folder
    ├── narrator000002/
    ├── ...
    └── narrator000600/          # 600 folders total (30,000 images)
```

### Expected Results

For 30,000 images (3 GPUs):
- **Infographic JSON files**: 600 files (30,000 / 50)
- **Wiki JSON files**: 600 files  
- **BizGen output folders**: 600 folders in `squad_v2/narrator*/`
- **Total PNG images**: 30,000

Each file contains 50 entries, matching the chunk_size in the scripts.

### Customizing Dataset Name

You can change the output folder name by modifying the `DATASET_NAME` variable in the pipeline script:

```bash
# In scripts/run_narrator_pipeline.sh
DATASET_NAME="my_custom_dataset"  # Will create output/my_custom_dataset/narrator*/
```

Or pass it directly when using inference.py:

```bash
python inference.py \
    --ckpt_dir checkpoints/lora/infographic \
    --wiki_dir ../create_data/output/narrator_format/ \
    --subset 0:100 \
    --dataset_name "my_custom_dataset"
```