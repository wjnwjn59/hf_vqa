# OCR Filter Script

A Python script to filter infographic images based on Jaccard similarity between expected text content (from JSON metadata) and actual OCR-detected text.

## Overview

This script uses **Jaccard similarity** to measure text content overlap between JSON metadata and OCR results. The Jaccard similarity coefficient is calculated as the intersection of word sets divided by their union, providing a value between 0.0 (no similarity) and 1.0 (perfect match). Images with similarity below the specified threshold are filtered out as potentially having poor text quality or mismatched content.

## Requirements

- Python 3.8+
- PaddleOCR
- OpenCV
- NumPy

## Installation

1. Activate your conda environment:
```bash
conda activate thinh_wiki
```

2. Install dependencies:
```bash
pip install paddlepaddle paddleocr opencv-python numpy
```
or
```bash
pip install -r src/data/ocr/requirements.txt
```

3. Set Python path:
```bash
export PYTHONPATH="./:$PYTHONPATH"
```

## Usage

### Basic Usage

```bash
python src/data/ocr/ocr_filter.py --start-id 1 --end-id 16
```

### Advanced Usage

```bash
python src/data/ocr/ocr_filter.py \
    --start-id 1 \
    --end-id 200 \
    --threshold 0.3 \
    --images-dir "/home/thinhnp/hf_vqa/src/data/bizgen/output/test" \
    --bizgen-dir "/home/thinhnp/hf_vqa/src/data/create_data/output/test_font" \
    --output-dir "src/data/create_data/output/ocr_filter"
```

## Arguments

- `--start-id`: Start image ID (default: 1)
- `--end-id`: End image ID (default: auto-detect from images)
- `--threshold`: Jaccard similarity threshold for filtering (default: 0.3)
- `--images-dir`: Directory containing images to process
- `--bizgen-dir`: Directory containing bizgen_format JSON files
- `--output-dir`: Output directory for filtered results

## Logic

The script uses **Jaccard similarity** to compare text content:

1. **Text Extraction**: Extracts text from JSON metadata and performs OCR on images
2. **Jaccard Calculation**: Computes similarity as `|intersection| / |union|` of word sets
3. **Filtering**: Images with Jaccard similarity below threshold are marked for filtering

## Output

The script generates a JSON file (`filtered_images.json`) containing:
- Metadata about the filtering process
- List of filtered images with detailed analysis (we use this list to remove low-quality images)
- Similarity scores and comparison metrics

## Example Output

```json
{
  "metadata": {
    "total_images_processed": 16,
    "total_images_filtered": 8,
    "filter_threshold": 0.3,
    "filter_ratio": 0.5
  },
  "filtered_images": [
    {
      "image_id": 1,
      "image_filename": "1.png",
      "json_text_count": 8,
      "ocr_text_count": 50,
      "text_similarity_ratio": 0.245,
      "reason": "Jaccard similarity (0.245) between JSON and OCR texts is below threshold (0.300)"
    }
  ]
}
```