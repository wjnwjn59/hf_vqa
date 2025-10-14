# Wikipedia Setup

## Symbolic Links or Download Checkpoints

```bash
ln -s /mnt/VLAI_data/BizGen/checkpoints ./src/data/bizgen/
```

```bash
gdown ...
```

## Download the Wiki Data (Only once)

```bash
conda env create -f ./src/data/create_data/wiki.yaml
conda activate wiki

export PYTHONPATH="./:$PYTHONPATH"

python src/data/create_data/wikipedia/download_wikipedia.py \
    --output_dir ./src/data/create_data/wikipedia
```

## Process the Wiki Data (Only once)

```bash
python src/data/create_data/wikipedia/extract_wikipedia_full.py \
    --dataset_path ./src/data/create_data/wikipedia/wikipedia_en_20231101 \
    --output_path ./src/data/create_data/wikipedia/wikipedia_processed \
    --min_words 1024 \
    --max_samples 232000 \
    --save_format json
```

## Process the Bboxes (Don't need to run)

Run this script to extract the bounding boxes from the Wikipedia data. This step may take a while.

```bash
cd ./src/data/create_data/qwen/

conda activate wiki
python extract_bboxes.py
```

# Create BizGen Data

## Create the Summaries

```bash
conda activate wiki

CUDA_VISIBLE_DEVICES=0 python src/data/create_data/qwen/generate_wikipedia_summary.py \
    --model_name "unsloth/Qwen3-8B" \
    --input_data "./src/data/create_data/wikipedia/wikipedia_processed.json" \
    --template_path "./src/prompts/summary.jinja" \
    --start-wiki 0 \
    --end-wiki 25800 \
    --batch_size 8
```

## Using Qwen to generate the infographic data

```bash
conda activate wiki

CUDA_VISIBLE_DEVICES=0 python src/data/create_data/qwen/generate_infographic_data.py \
    --model_name "unsloth/Qwen3-8B" \
    --input_data "./src/data/create_data/output/summarize" \
    --template_path "./src/prompts/bizgen_full.jinja" \
    --start-wiki 0 \
    --end-wiki 25800 \
    --batch_size 8 
```

## Merge the Bboxes with the generated data

```bash
conda activate wiki

python src/data/create_data/qwen/merge_infographic_bboxes.py \
    --extracted-bboxes ./src/data/create_data/qwen/extracted_bboxes.json \
    --infographic-dir ./src/data/create_data/output/infographic \
    --color-idx ./src/data/create_data/qwen/glyph/color_idx.json \
    --font-idx ./src/data/create_data/qwen/glyph/font_idx.json \
    --start-wiki 0 \
    --end-wiki 232000 \
    --seed 42
```

## Split the BBoxes with the generated data

```bash
conda activate wiki

python src/data/create_data/qwen/extract_infographic.py \
    --infographic-dir ./src/data/create_data/output/infographic \
    --color-idx ./src/data/create_data/qwen/glyph/color_idx.json \
    --font-idx ./src/data/create_data/qwen/glyph/font_idx.json \
    --start-wiki 0 \
    --end-wiki 0 \
    --seed 42
```

## Run the script to create the data

```bash
conda env create -f ./src/data/bizgen/wiki.yaml
conda activate bizgen

cd ./src/data/bizgen/

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --ckpt_dir checkpoints/lora/infographic \
    --wiki_dir ../create_data/output/bizgen_format/ \
    --subset 0:516
```

# Run OCR filter

```bash
conda activate wiki
export PYTHONPATH="./:$PYTHONPATH"

python src/data/ocr/ocr_filter.py \
    --images-dir "src/data/create_data/output/infographic_data_no_parse" \
    --start-id 1 \
    --end-id 232000
```

# Create VQA data

```bash
conda activate wiki
export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python src/data/vqa/generate_vqa_data.py \
    --model_name "unsloth/Qwen2-VL-7B-Instruct" \
    --images_dir "./src/data/bizgen/output/subset_0_516" \
    --template_path "./src/prompts/vqg.jinja" \
    --output_path "./src/data/vqa/vqa_data.json" \
    --num_questions 25800 \
    --batch_size 4
```