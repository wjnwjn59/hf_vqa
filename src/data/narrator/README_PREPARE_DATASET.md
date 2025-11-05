# Dataset Preparation Script

## Overview

The `prepare_dataset.py` script reorganizes generated NARRATOR VQA infographic data into a structured dataset format compatible with SQuAD v2.

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
├── train_annotations.json      # SQuAD v2 format train annotations (JSONL)
└── val_annotations.json        # SQuAD v2 format val annotations (JSONL)
```

## Usage

### For Training Data

```bash
python src/data/narrator/prepare_dataset.py \
    --wiki-dir src/data/narrator/wiki \
    --image-source-dir src/data/bizgen/output \
    --dataset-name squad_v2 \
    --output-dir /home/thinhnp/hf_vqa/dataset \
    --type train
```

### For Validation Data

```bash
python src/data/narrator/prepare_dataset.py \
    --wiki-dir src/data/narrator/wiki_val \
    --image-source-dir src/data/bizgen/output \
    --dataset-name squad_v2_val \
    --output-dir /home/thinhnp/hf_vqa/dataset \
    --type val
```

## Arguments

- `--wiki-dir`: Source directory containing wiki*.json files (default: `src/data/narrator/wiki`)
- `--image-source-dir`: Base directory with generated images (default: `src/data/bizgen/output`)
- `--dataset-name`: Dataset folder name in image source directory (default: `squad_v2`)
- `--output-dir`: Target dataset directory (default: `/home/thinhnp/hf_vqa/dataset`)
- `--type`: Annotation type - either `train` or `val` (required)
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
  "qa_pairs": [...]
}
```

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

## How It Works

1. Reads all `wiki*.json` files from the wiki directory
2. Loads reasoning data from `generated_reasonings.jsonl` and indexes it
3. For each entry with index N:
   - Copies image from `{image-source-dir}/{dataset-name}/narratorXXXXXX/{N}.png` to `{output-dir}/images/{N}.png`
   - Creates `{output-dir}/templates/{N}.json` with template data
   - Creates `{output-dir}/qas/{N}.json` with:
     - Original QA pairs with reasoning
     - Generated QA pairs with reasoning
   - Generates lightweight annotation metadata entries
4. Writes all annotations to `{output-dir}/{type}_annotations.jsonl` (JSONL format)

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

## Example Output

After processing 50 images from `wiki000001.json`:

```
Dataset Summary:
- Total entries found: 50
- Successfully processed: 50
- Images copied: 50
- Template files created: 50
- QAS files created: 50
- Annotation entries: 574 (multiple QA pairs per image)
```

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

