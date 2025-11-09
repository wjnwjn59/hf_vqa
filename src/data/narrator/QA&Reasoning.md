# Data Generation Pipeline

This document outlines the two-step pipeline used to generate question-answer pairs and their corresponding reasoning chains.

**The workflow must be run in sequence:**

1.  **Run `generate_qas.py`**: This script enriches the source `wiki*.json` files by generating and adding new Q\&A pairs.
2.  **Run `generate_reasoning.py`**: This script reads the *enriched* `wiki*.json` files (from Step 1) and generates detailed reasoning for *all* Q\&A pairs (both original and new), saving the final output to a separate `.jsonl` file.

## 🚀 Backend Support

Both scripts support selectable inference backends via the `--backend` argument:

  * **`--backend qwen`** (Default): Uses a locally loaded Qwen model (via vLLM).
      * Specify the model path with `--model_name`.
  * **`--backend gpt`**: Uses the OpenAI API.
      * Specify the model with `--openai_model` (e.g., "gpt-4o").
      * Requires an API key via `--openai_api_key` or the `OPENAI_API_KEY` environment variable.

-----

## 1\. Step 1: QA Generation (`generate_qas.py`)

This script's purpose is to enrich the dataset by generating *new* question-answer pairs based on the existing infographic data.

  * **Input**: Reads all `wiki*.json` files from the directory specified by `--layout_dir`.
  * **Output**: Modifies the `wiki*.json` files **in-place** by adding a new `generated_qa_pairs` list to each item.

### Usage Examples

**Using Qwen (default):**

```bash
python src/data/narrator/generate_qas.py \
    --model_name "/path/to/your/Qwen_Qwen3-8B" \
    --layout_dir "/path/to/your/data/wiki" \
    --k_value 3
```

**Using GPT (OpenAI):**

```bash
python src/data/narrator/generate_qas.py \
    --backend gpt \
    --openai_model "gpt-4o" \
    --layout_dir "/path/to/your/data/wiki" \
    --k_value 3
```

-----

## 2\. Step 2: Reasoning Generation (`generate_reasoning.py`)

This script's purpose is to generate a detailed, step-by-step logical reasoning chain for *every* QA pair in the dataset.

  * **Input**: Reads all `wiki*.json` files from the `--layout_dir` (which *must* have already been processed by Step 1).
  * **Output**: Creates a new **JSON Lines** (`.jsonl`) file at the path specified by `--output_file_path`.
      * Each line in this file is a JSON object containing the raw `generated_reasoning` JSON from the model, plus **four** automatically generated natural language versions for ablation studies:
        1.  `reasoning_full`: The complete reasoning with all details.
        2.  `reasoning_no_bbox`: Reasoning *without* bounding box coordinates.
        3.  `reasoning_no_spatial`: Reasoning *without* spatial context (e.g., "below the title").
        4.  `reasoning_no_bbox_no_spatial`: Reasoning *without* either bbox or spatial info.

### Usage Examples

**Using Qwen (default):**

```bash
python src/data/narrator/generate_reasoning.py \
    --model_name "/path/to/your/Qwen_Qwen3-8B" \
    --layout_dir "/path/to/your/data/wiki" \
    --output_file_path "/path/to/your/data/generated_reasonings.jsonl"
```

**Using GPT (OpenAI):**

```bash
python src/data/narrator/generate_reasoning.py \
    --backend gpt \
    --openai_model "gpt-4o" \
    --layout_dir "/path/to/your/data/wiki" \
    --output_file_path "/path/to/your/data/generated_Gpt_reasonings.jsonl"
```