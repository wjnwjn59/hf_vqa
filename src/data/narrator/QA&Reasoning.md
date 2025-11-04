# Data Generation Pipeline Overview

This document outlines the two-step pipeline used to generate question-answer pairs and their corresponding reasoning chains. The pipeline consists of two main scripts that must be run sequentially.

**The workflow is as follows:**

1.  **Run `qa_generation.py`**: This script reads the base JSON files, generates new Q\&A pairs using a specified backend (Qwen or GPT), and overwrites the original JSON files to include this new data.
2.  **Run `reasoning_generation.py`**: This script reads the *enriched* JSON files (from Step 1) and generates detailed reasoning for *all* Q\&A pairs (both original and newly generated), saving the final output to a separate `.jsonl` file.

-----

## 🚀 Backend Support

Both scripts now support selectable inference backends via the `--backend` argument:

  * **`--backend qwen`** (Default): Uses a locally loaded Qwen3-8B model.
      * Specify the model path with `--model_name`.
  * **`--backend gpt`**: Uses the OpenAI API (e.g., GPT-4o).
      * Specify the model with `--openai_model`.
      * Requires an API key via `--openai_api_key` or the `OPENAI_API_KEY` environment variable.

-----

## 1\. QA Generation (`qa_generation.py`)

This script's purpose is to enrich the dataset by generating *new* question-answer pairs based on the existing infographic data.

### 📜 Input

  * **Source Directory**: `hf_vqa/src/data/wiki/` (configurable via `--layout_dir`)
  * **Input Files**: Reads all `wiki*.json` files from the source directory.
  * **Data Structure**: For each JSON object (item) in a file, it uses:
      * `layers_all` and `full_image_caption`: As the visual context for generation.
      * `original_qa_pairs`: As few-shot examples to guide the model's output format.

### 💾 Output

  * **Output Target**: **This script modifies the input files in-place.** It reads a `wiki*.json` file, modifies its content in memory, and then overwrites the original file.
  * **Data Structure**: It adds a new key, `generated_qa_pairs`, to each JSON object (item) inside the `wiki*.json` files.

**Example `wiki000001.json` (Item 1 after running):**

```json
[
  {
    "index": 1,
    "layers_all": [...],
    "full_image_caption": "...",
    "original_qa_pairs": [
      {"question": "Q_orig_1", "answers": ...},
      {"question": "Q_orig_2", "answers": ...}
    ],
    "generated_qa_pairs": [
      {"question": "Q_new_1", "answer": "A_new_1"},
      {"question": "Q_new_2", "answer": "A_new_2"}
    ]
  }
]
```

### 🏃 Usage Examples

**Using Qwen (default):**

```bash
python qa_generation.py \
    --model_name "/mnt/dataset1/pretrained_fm/Qwen_Qwen3-8B" \
    --layout_dir "/home/binhdt/hf_vqa/src/data/wiki/" \
    --k_value 3
```

**Using GPT (OpenAI):**

```bash
python qa_generation.py \
    --backend gpt \
    --openai_model "gpt-4o" \
    --openai_api_key "sk-..." \
    --layout_dir "/home/binhdt/hf_vqa/src/data/wiki/" \
    --k_value 3
```

-----

## 2\. Reasoning Generation (`reasoning_generation.py`)

This script's purpose is to generate a detailed, step-by-step logical reasoning chain for *every* question-answer pair in the dataset.

### 📜 Input

  * **Source Directory**: `hf_vqa/src/data/wiki/` (configurable via `--layout_dir`)
  * **Input Files**: Reads all `wiki*.json` files from the source directory (which *must* have already been processed by `qa_generation.py`).
  * **Data Structure**: For each JSON object (item) in a file, it reads:
      * `layers_all` and `full_image_caption`: As the visual context for reasoning.
      * `original_qa_pairs`: The list of original Q\&A pairs.
      * `generated_qa_pairs`: The list of new Q\&A pairs (added by Script 1).

### 💾 Output

  * **Output File**: `../narrator/generated_reasonings.jsonl` (configurable via `--output_file_path`)
  * **Output Format**: A **JSON Lines** (`.jsonl`) file. Each line is a complete JSON object representing the reasoning for one Q\&A pair.
  * **Data Structure (per line)**:

<!-- end list -->

```json
{
  "wiki_id": "000001",
  "layout_index": 1,
  "squad_id": "56be85543aeaaa14008c9063", // or "gen_1_1" for generated QAs
  "question": "When did Beyonce start becoming popular?",
  "ground_truth_answer": "in the late 1990s",
  "generated_reasoning": {
    "understand": {
      "analysis": "...",
      "relevant_elements": [...]
    },
    "think": {
      "evidence_array": [...],
      "logical_reasoning": "..."
    },
    "answer": "in the late 1990s"
  },
  "merged_reasoning": "The question asks for when Beyoncé became popular. The infographic contains a text block [U1]... Therefore, the answer is in the late 1990s."
}
```

### 🏃 Usage Examples

**Using Qwen (default):**

```bash
python reasoning_generation.py \
    --model_name "/mnt/dataset1/pretrained_fm/Qwen_Qwen3-8B" \
    --layout_dir "/home/binhdt/hf_vqa/src/data/wiki/" \
    --output_file_path "../narrator/generated_reasonings.jsonl"
```

**Using GPT (OpenAI):**

```bash
python reasoning_generation.py \
    --backend gpt \
    --openai_model "gpt-4o" \
    --openai_api_key "sk-..." \
    --layout_dir "/home/binhdt/hf_vqa/src/data/wiki/" \
    --output_file_path "../narrator/generated_reasonings_gpt.jsonl"
```