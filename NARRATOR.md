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

## Using Qwen to generate layout

```python
conda env create -f ./src/data/create_data/bizgen/wiki.yaml
conda activate wiki

export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python src/data/narrator/generate_narrator_with_bbox.py \
    --model_name "unsloth/Qwen3-8B" \
    --input_data "/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl" \
    --stage_1 "./src/prompts/content_des_stage_1.jinja" \
    --stage_2 "./src/prompts/content_des_stage_2.jinja" \
    --start 1 \
    --end 10 \
    --batch_size 8
```

## Create bizgen data

```bash
conda activate bizgen

cd ./src/data/bizgen/

python inference.py \
    --ckpt_dir checkpoints/lora/infographic \
    --wiki_dir ../create_data/output/narrator_format/ \
    --subset 1:10 \
    --device cuda:2 \
    --dataset_name squad_v2
```

## Run OCR Filter

```bash
conda activate wiki
export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python src/data/ocr/ocr_filter.py \
    --images-dir "./src/data/bizgen/output/squad_v2" \
    --bizgen-dir "./src/data/create_data/output/narrator_format" \
    --output-dir "./src/data/bizgen/output/squad_v2/ocr_result" \
    --threshold 0.5
```

## Running Full Pipeline on Multiple GPUs

### NARRATOR v2 Pipeline (3-Stage Generation)

To run the complete NARRATOR v2 pipeline (using 3-stage generation) on 3 GPUs automatically:

```bash
# Make scripts executable
chmod +x scripts/run_narrator_v2_pipeline.sh
chmod +x scripts/run_narrator_v2_parallel.sh

# Run on all GPUs in parallel
bash scripts/run_narrator_v2_parallel.sh
```

This will:
- **GPU 0**: Process subsets 1-200 (10,000 images)
- **GPU 1**: Process subsets 201-400 (10,000 images)  
- **GPU 2**: Process subsets 401-600 (10,000 images)

**Total**: 30,000 images, 600 subsets

### Original Pipeline (Single-Stage Generation)

To run the original pipeline on 3 GPUs automatically:

```bash
# Make scripts executable
chmod +x scripts/run_narrator_pipeline.sh
chmod +x scripts/run_narrator_parallel.sh

# Run on all GPUs in parallel
bash scripts/run_narrator_parallel.sh
```

This will:
- **GPU 0**: Process files 1-200 (10,000 images)
- **GPU 1**: Process files 201-400 (10,000 images)  
- **GPU 2**: Process files 401-600 (10,000 images)

**Total**: 30,000 images, 600 files

### Single GPU Usage

#### NARRATOR v2 Pipeline (3-Stage)

To run the v2 pipeline on a single GPU with custom range:

```bash
# Syntax: bash scripts/run_narrator_v2_pipeline.sh <GPU_ID> <START_SUBSET> <END_SUBSET>

# GPU 0: process 10,000 images (subsets 1-200)
bash scripts/run_narrator_v2_pipeline.sh 0 1 201

# GPU 1: process next 10,000 images (subsets 201-400)
bash scripts/run_narrator_v2_pipeline.sh 1 201 401

# GPU 2: process final 10,000 images (subsets 401-600)
bash scripts/run_narrator_v2_pipeline.sh 2 401 601
```

#### Original Pipeline (Single-Stage)

To run the original pipeline on a single GPU with custom range:

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

#### NARRATOR v2 Pipeline (3-Stage)

Each GPU runs 3 steps sequentially:

1. **Generate 3-Stage Infographic Data** (`generate_stage_infographic.py`)
   - Uses Qwen3-8B to generate infographic descriptions through 3 stages
   - Stage 1: Sentence summarization
   - Stage 2: Figure idea generation
   - Stage 3: Final caption composition
   - Input: Squad v2 dataset
   - Output: `infographic*.json` files (50 entries per file) in `infographic_v2/`

2. **Merge Bounding Boxes** (`merge_stage_narrator.py`)
   - Merges generated descriptions with bbox data
   - Output: `wiki*.json` files in narrator format in `narrator_format_v2/`

3. **Generate Images** (BizGen `inference.py`)
   - Creates actual infographic images
   - Output: PNG images in `output/squad_v2_new/narrator*/` folders

#### Original Pipeline (Single-Stage)

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

#### NARRATOR v2 Pipeline

```bash
# Watch all GPUs
tail -f logs/narrator_v2_pipeline/*_<timestamp>.log

# Watch specific GPU
tail -f logs/narrator_v2_pipeline/gpu0_*.log
tail -f logs/narrator_v2_pipeline/gpu1_*.log
tail -f logs/narrator_v2_pipeline/gpu2_*.log

# Monitor GPU usage
nvidia-smi -l 1
# or
nvtop
```

#### Original Pipeline

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

#### NARRATOR v2 Pipeline

Edit `scripts/run_narrator_v2_parallel.sh` to customize GPU ranges:

```bash
GPU_CONFIGS=(
    "0 1 201"       # GPU 0: subsets 1-200 (10,000 images)
    "1 201 401"     # GPU 1: subsets 201-400 (10,000 images)
    "2 401 601"     # GPU 2: subsets 401-600 (10,000 images)
)
```

#### Original Pipeline

Edit `scripts/run_narrator_parallel.sh` to customize GPU ranges:

```bash
GPU_CONFIGS=(
    "0 0 10000"       # GPU 0: images 0-9999
    "1 10000 20000"   # GPU 1: images 10000-19999
    "2 20000 30000"   # GPU 2: images 20000-29999
)
```

### Output Structure

#### NARRATOR v2 Pipeline

```
src/data/create_data/output/
├── infographic_v2/
│   ├── infographic000001.json    # 50 entries: IDs 1-50
│   ├── infographic000002.json    # 50 entries: IDs 51-100
│   └── ...
│   └── infographic000600.json    # 50 entries: IDs 29951-30000
├── narrator_format_v2/
│   ├── wiki000001.json           # Merged format
│   ├── wiki000002.json
│   └── ...
│   └── wiki000600.json

src/data/bizgen/output/
└── squad_v2_new/               # Dataset output folder
    ├── narrator000001/         # 50 images per folder
    ├── narrator000002/
    ├── ...
    └── narrator000600/         # 600 folders total (30,000 images)
```

#### Original Pipeline

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

#### NARRATOR v2 Pipeline

For 30,000 images (3 GPUs):
- **Infographic v2 JSON files**: 600 files (30,000 / 50)
- **Wiki JSON files**: 600 files  
- **BizGen output folders**: 600 folders in `squad_v2_new/narrator*/`
- **Total PNG images**: 30,000

#### Original Pipeline

For 30,000 images (3 GPUs):
- **Infographic JSON files**: 600 files (30,000 / 50)
- **Wiki JSON files**: 600 files  
- **BizGen output folders**: 600 folders in `squad_v2/narrator*/`
- **Total PNG images**: 30,000

Each file contains 50 entries, matching the chunk_size in the scripts.

### Customizing Dataset Name

#### NARRATOR v2 Pipeline

You can change the output folder name by modifying the `DATASET_NAME` variable in the v2 pipeline script:

```bash
# In scripts/run_narrator_v2_pipeline.sh
DATASET_NAME="my_custom_dataset_v2"  # Will create output/my_custom_dataset_v2/narrator*/
```

Or pass it directly when using inference.py:

```bash
python inference.py \
    --ckpt_dir checkpoints/lora/infographic \
    --wiki_dir ../create_data/output/narrator_format_v2/ \
    --subset 0:100 \
    --dataset_name "my_custom_dataset_v2"
```

#### Original Pipeline

You can change the output folder name by modifying the `DATASET_NAME` variable in the original pipeline script:

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