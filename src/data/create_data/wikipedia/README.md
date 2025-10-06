# Wikipedia Data Processing

This folder contains tools for downloading and processing English Wikipedia data from Hugging Face Datasets.

## Overview

The Wikipedia processing pipeline consists of 2 main steps:
1. **Download dataset**: Use `download_wikipedia.py` to download Wikipedia from Hugging Face
2. **Extract and filter content**: Use `extract_wikipedia_full.py` to filter and clean the data

## Installation

Install the required dependencies:

```bash
cd /home/thinhnp/hf_vqa/src/data/create_data/wikipedia
pip install -r requirements.txt
```

## Usage Guide

### Step 1: Download Wikipedia Dataset

```bash
python download_wikipedia.py \
    --output_dir ./src/data/create_data/wikipedia \
    --cache_dir /path/to/cache  # Optional
```

**Parameters:**
- `--output_dir`: Directory to save dataset (default: `./src/data/create_data/wikipedia`)
- `--cache_dir`: Cache directory for Hugging Face (optional)

**Result:** Dataset will be saved to `wikipedia_en_20231101/`

### Step 2: Extract and process data

```bash
python extract_wikipedia_full.py \
    --dataset_path ./src/data/create_data/wikipedia/wikipedia_en_20231101 \
    --output_path ./src/data/create_data/wikipedia/wikipedia_full_processed \
    --min_words 1024 \
    --max_samples 232000 \
    --save_format json
```

Because our infographic generation model will remove some image that not meet OCR requirement, increase the `--max_samples` to 250000 to ensure we have enough samples after filtering.

**Parameters:**
- `--dataset_path`: Path to downloaded dataset (default: `./src/data/create_data/wikipedia/wikipedia_en_20231101`)
- `--output_path`: Path to save results (default: `./src/data/create_data/wikipedia/wikipedia_full_processed`)
- `--min_words`: Minimum words per article (default: 1024)
- `--max_samples`: Maximum number of articles (default: 1000)
- `--save_format`: File format (`json` or `jsonl`, default: `json`)

## Data Filtering Features

### Remove unwanted content
- **History and politics topics**: Automatically excludes articles about history, wars, politics
- **Reference sections**: Removes "References", "See also", "External links", "Bibliography" sections
- **Short articles**: Only keeps articles with at least `min_words` words

### Excluded keywords
```
history, politics, wars, battles, elections, government, military, 
empires, revolutions, conflicts, diplomacy, presidents, monarchy, etc.
```

## Output Data Structure

```json
{
  "id": "article_id",
  "title": "Article Title", 
  "text": "Full article content without references...",
  "categories": ["category1", "category2"]
}
```

## Complete Usage Example

```bash
# Step 1: Download dataset
python download_wikipedia.py

# Step 2: Process data
python extract_wikipedia_full.py \
    --max_samples 5000 \
    --min_words 500
```

## Generated Files

- `wikipedia_en_20231101/`: Original dataset from Hugging Face
- `wikipedia_full_processed.json`: Processed and filtered data
- `wikipedia_processed.json`: Result file (if any)

## Notes

- Wikipedia dataset is quite large (~20GB), ensure sufficient disk space
- Download process may take several hours depending on network speed
- Use `--cache_dir` to avoid re-downloading when running multiple times
- Increase `--max_samples` to process more articles (will take more time)