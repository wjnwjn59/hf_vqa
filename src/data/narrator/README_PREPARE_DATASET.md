# Dataset Preparation Script

## Overview

The `prepare_dataset.py` script reorganizes generated NARRATOR VQA infographic data into a structured dataset format compatible with SQuAD v2.

## Key Features

- Loads and integrates reasoning data with 4 reasoning types:
  - `reasoning_full`: Full reasoning with spatial context and bounding boxes
  - `reasoning_no_bbox`: Reasoning without bounding box coordinates
  - `reasoning_no_spatial`: Reasoning without spatial descriptions
  - `reasoning_short`: Concise reasoning summary
- Handles unanswerable questions (empty answers → "unanswerable")
- Generates TSV files for easy data loading
- Creates lightweight JSONL annotation files with metadata

## Output Structure

```
/home/thinhnp/hf_vqa/dataset/
├── images/                      # All images with simple numeric names
│   ├── 1.png
│   ├── 2.png
│   └── ...
├── templates/                   # Template files (layers, captions, bboxes)
│   ├── 1.json
│   ├── 2.json
│   └── ...
├── qas/                        # QA pairs for each image
│   ├── 1.json
│   ├── 2.json
│   └── ...
├── train_annotations.jsonl     # SQuAD v2 format train annotations (JSONL)
├── train_data.tsv              # TSV format train data
├── val_annotations.jsonl       # SQuAD v2 format val annotations (JSONL)
└── val_data.tsv                # TSV format val data
```

## Usage

### For Training Data (with reasoning)

```bash
python src/data/narrator/prepare_dataset.py \
    --wiki-dir src/data/narrator/wiki \
    --image-source-dir src/data/bizgen/output \
    --dataset-name squad_v2_train \
    --output-dir /home/thinhnp/hf_vqa/dataset \
    --reasoning-file /home/thinhnp/hf_vqa/src/data/narrator/generated_gpt5_reasonings_final.jsonl \
    --type train
```

### For Validation Data (without reasoning)

```bash
python src/data/narrator/prepare_dataset.py \
    --wiki-dir src/data/narrator/wiki_val \
    --image-source-dir src/data/bizgen/output \
    --dataset-name squad_v2 \
    --output-dir /home/thinhnp/hf_vqa/dataset \
    --type val
```

**Note:** If `--reasoning-file` is not provided or the file does not exist, QA pairs will be created without reasoning fields. This is useful when you want to generate the dataset structure first and add reasoning later.

## Arguments

- `--wiki-dir`: Source directory containing wiki*.json files (default: `src/data/narrator/wiki`)
- `--image-source-dir`: Base directory with generated images (default: `src/data/bizgen/output`)
- `--dataset-name`: Dataset folder name in image source directory (default: `squad_v2`)
- `--output-dir`: Target dataset directory (default: `/home/thinhnp/hf_vqa/dataset`)
- `--type`: Annotation type - either `train` or `val` (required)
- `--reasoning-file`: Path to reasoning JSONL file (default: `/home/thinhnp/hf_vqa/src/data/narrator/generated_gpt5_reasonings_final.jsonl`, optional - if not provided or file doesn't exist, reasoning fields will be empty)
- `--squad-file`: Path to original SQuAD file for reference (optional)

## File Formats

### Template File (`templates/{id}.json`)

```json
{
  "infographic_id": 1,
  "full_image_caption": "...",
  "layers_all": [...],
  "original_bbox_index": 36
}
```

### QAS File (`qas/{id}.json`)

```json
{
  "infographic_id": 1,
  "context": "...",
  "original_qa_pairs": [
    {
      "id": "56be85543aeaaa14008c9063",
      "question": "When did Beyonce start becoming popular?",
      "answer": "in the late 1990s",
      "is_unanswerable": false,
      "reasoning": {
        "generated_reasoning": {...},
        "merged_reasoning": "...",
        "reasoning_full": "Full reasoning with spatial context and bounding boxes...",
        "reasoning_no_bbox": "Reasoning without coordinates...",
        "reasoning_no_spatial": "Reasoning without spatial descriptions...",
        "reasoning_short": "Short summary..."
      }
    }
  ],
  "generated_qa_pairs": [
    {
      "squad_id": "gen_1_1",
      "question": "What is shown in the image?",
      "answer": "unanswerable",
      "is_unanswerable": true,
      "reasoning": {
        "generated_reasoning": {...},
        "merged_reasoning": "...",
        "reasoning_full": "...",
        "reasoning_no_bbox": "...",
        "reasoning_no_spatial": "...",
        "reasoning_short": "..."
      }
    }
  ]
}
```

**Note:** The `reasoning` field is only present if reasoning data is available. If `--reasoning-file` is not provided, QA pairs will not have the `reasoning` field.

### Annotation File (Lightweight Metadata JSONL format)

Each line is a lightweight metadata entry that references the full QAS data:

```json
{
  "id": "56be85543aeaaa14008c9063",
  "image_id": 1,
  "qas_file": "qas/1.json",
  "qa_type": "original",
  "qa_index": 0
}
```

For generated QA pairs:
```json
{
  "id": "gen_1_1",
  "image_id": 1,
  "qas_file": "qas/1.json",
  "qa_type": "generated",
  "qa_index": 0
}
```

This format reduces file size by 94% compared to storing full content in annotations.

### TSV File (`{type}_data.tsv`)

Tab-separated values file for easy data loading with format: `index\tanswer\tquestion\timage_path`

```tsv
index	answer	question	image_path
1	in the late 1990s	When did Beyonce start becoming popular?	/home/thinhnp/hf_vqa/dataset/images/1.png
1	singing and dancing	What areas did Beyonce compete in when she was growing up?	/home/thinhnp/hf_vqa/dataset/images/1.png
2	unanswerable	What is the color of the sky?	/home/thinhnp/hf_vqa/dataset/images/2.png
```

**Features:**
- Includes both original and generated QA pairs
- Uses absolute paths for image locations
- Unanswerable questions have answer = "unanswerable"
- For multiple answers, uses the shortest one
- Tabs and newlines in text are replaced with spaces

## How It Works

1. Reads all `wiki*.json` files from the wiki directory
2. Loads reasoning data from reasoning file (if provided) and indexes it by (wiki_id, layout_index, squad_id)
3. For each entry with index N:
   - Copies image from `{image-source-dir}/{dataset-name}/narratorXXXXXX/{N}.png` to `{output-dir}/images/{N}.png`
   - Creates `{output-dir}/templates/{N}.json` with template data
   - Creates `{output-dir}/qas/{N}.json` with:
     - Original QA pairs with reasoning (if available) and unanswerable handling
     - Generated QA pairs with reasoning (if available) and unanswerable handling
     - Each QA includes 6 reasoning fields: generated_reasoning, merged_reasoning, reasoning_full, reasoning_no_bbox, reasoning_no_spatial, reasoning_short
   - Generates lightweight annotation metadata entries
   - Collects TSV entries with shortest answers
4. Writes all annotations to `{output-dir}/{type}_annotations.jsonl` (JSONL format)
5. Writes TSV data to `{output-dir}/{type}_data.tsv` with absolute image paths

## Loading Annotation Data

Since annotations now only contain metadata, you need to load the full QAS data separately. See `example_load_annotation.py` for a complete example:

```python
import json
from pathlib import Path

# Load QAS cache
qas_cache = {}
qas_dir = Path(dataset_dir) / "qas"
for qas_file in qas_dir.glob("*.json"):
    image_id = int(qas_file.stem)
    with open(qas_file, 'r') as f:
        qas_cache[image_id] = json.load(f)

# Read annotations
with open("train_annotations.jsonl", 'r') as f:
    for line in f:
        annotation = json.loads(line)
        
        # Get full QA data
        qas_data = qas_cache[annotation["image_id"]]
        if annotation["qa_type"] == "original":
            qa = qas_data["original_qa_pairs"][annotation["qa_index"]]
        else:
            qa = qas_data["generated_qa_pairs"][annotation["qa_index"]]
        
        # Now you have: qa["question"], qa["answers"], qa["reasoning"], etc.
```

## Notes

- Images are named with simple numbers: `1.png`, `2.png`, etc.
- Template and QAS files use the same numbering
- Each wiki file contains 50 entries:
  - `wiki000001.json` → indices 1-50
  - `wiki000002.json` → indices 51-100
  - etc.
- The script automatically skips entries with missing images
- Annotations follow JSONL format (one JSON object per line)
- Each QA pair becomes a separate entry in the annotations file
- The `image_id` field links each QA to its corresponding image

### Reasoning Data (Optional)

- If `--reasoning-file` is not provided or the file doesn't exist:
  - QA pairs will be created **without** the `reasoning` field
  - All other functionality works normally
  - This allows you to generate dataset structure first and add reasoning later
- When reasoning file is provided:
  - The script loads and indexes reasoning by (wiki_id, layout_index, squad_id)
  - Matches reasoning to both original and generated QA pairs
  - Adds 6 reasoning fields: `generated_reasoning`, `merged_reasoning`, `reasoning_full`, `reasoning_no_bbox`, `reasoning_no_spatial`, `reasoning_short`

### Unanswerable Questions

- Empty or None answers are automatically converted to "unanswerable"
- Each QA pair has an `is_unanswerable` field (true/false)
- In TSV files, unanswerable questions have answer = "unanswerable"
- For multiple answers, the shortest non-empty answer is selected
- This follows SQuAD v2 format for unanswerable questions

## Workflow

1. Generate infographics using NARRATOR pipeline → creates `wiki*.json` files
2. Generate images using BizGen → creates images in `narrator*/` folders
3. Run this script to organize everything into dataset format
4. Use the dataset for training/evaluation

## Integration with Pipeline

This script is designed to work with the output of:

- `src/data/narrator/generate_infographic_data.py` → generates `infographic*.json`
- `src/data/narrator/merge_narrator_bboxes.py` → generates `wiki*.json`
- `src/data/bizgen/inference.py` → generates images in `narrator*/` folders

