# Summary of Changes to generate_stage_infographic.py

## Overview
Modified the script to implement keyword checking and retry logic with different seeds when keywords from QA pairs are not found in the generated full_image_caption.

## Key Changes Made

### 1. Enhanced Imports and Utilities
- Added `import random` for seed management
- Added `set_random_seed()` function to manage random seeds for retry attempts

### 2. Improved Keyword Handling
- Updated `extract_answer_keywords()` to preserve original case for exact matching
- Modified `check_keywords_in_caption()` to return both boolean result and list of found keywords
- Implemented case-insensitive matching while preserving original keyword formats

### 3. Enhanced process_sample() Function
**New Parameters:**
- `max_retries`: Maximum number of retry attempts (default: 2)

**New Logic:**
- Extract keywords from QA pairs for each context
- Retry loop: attempts up to `max_retries + 1` times (including first attempt)
- Different random seed for each retry attempt
- Check if keywords are present in `full_image_caption` after each generation
- Return `None` if keywords not found after all retries
- Enhanced error handling and logging

**New Output Fields:**
- `context`: Original context text
- `qa_pairs`: All QA pairs for this context
- `keywords`: Extracted keywords from answers
- `keywords_found`: List of keywords actually found in caption
- `retry_count`: Number of retries performed
- `final_seed`: Seed used for successful generation

### 4. Updated Data Processing
- Added `--max_retries` command line argument (default: 2)
- Enhanced file saving to handle `None` results (failed keyword checks)
- Updated statistics tracking to distinguish between:
  - Successful generations
  - Failed generations (errors)
  - Failed generations (keywords not found)

### 5. Improved File Handling
- Modified `save_chunk_to_file()` to filter out `None` results
- Enhanced logging to show keyword check failures
- Updated file index calculation to handle missing results

### 6. Enhanced Logging and Statistics
- Added keyword information in processing logs
- Show sample keywords from first context
- Detailed retry attempt logging
- Comprehensive final statistics including keyword failure counts

## Usage Examples

### Basic Usage (default 2 retries)
```bash
python generate_stage_infographic.py \
    --input_data /path/to/squad_v2_train.jsonl \
    --output_dir /path/to/output \
    --start 1 --end 5
```

### Custom retry count
```bash
python generate_stage_infographic.py \
    --input_data /path/to/squad_v2_train.jsonl \
    --output_dir /path/to/output \
    --max_retries 3 \
    --start 1 --end 5
```

### No retries (original behavior)
```bash
python generate_stage_infographic.py \
    --input_data /path/to/squad_v2_train.jsonl \
    --output_dir /path/to/output \
    --max_retries 0 \
    --start 1 --end 5
```

## Expected Behavior

1. **For each unique context:**
   - Extract all answer keywords from associated QA pairs
   - Generate infographic description through 3 stages
   - Check if keywords appear in `full_image_caption`
   - If keywords found: save result
   - If keywords not found: retry with different seed
   - After max retries: return `None` (not saved to output)

2. **Output files contain only successful generations** where keywords were found

3. **Statistics show:**
   - Total contexts processed
   - Successful generations (keywords found)
   - Failed generations (errors during processing)
   - Failed generations (keywords not found after retries)

## Example Input/Output

### Input (JSONL):
```json
{
  "id": "56be85543aeaaa14008c9063",
  "title": "Beyoncé", 
  "context": "Beyoncé rose to fame in the late 1990s as lead singer...",
  "question": "When did Beyonce start becoming popular?",
  "answers": {"text": ["in the late 1990s"], "answer_start": [269]}
}
```

### Output (JSON):
```json
{
  "id": "56be85543aeaaa14008c9063",
  "title": "Beyoncé",
  "context": "Beyoncé rose to fame in the late 1990s as lead singer...",
  "qa_pairs": [{"question": "When did Beyonce start becoming popular?", "answers": {"text": ["in the late 1990s"]}}],
  "keywords": ["in the late 1990s"],
  "keywords_found": ["in the late 1990s"],
  "retry_count": 0,
  "final_seed": 12345,
  "generated_infographic": {
    "full_image_caption": "The image shows an infographic about Beyoncé's rise to fame in the late 1990s..."
  },
  "success": true,
  "infographic_id": 1
}
```

This ensures that generated infographic captions contain the specific answer keywords needed for the VQA task.