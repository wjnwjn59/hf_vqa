# VQA Inference

## Overview

Run Vision-Language Model (VLM) inference on the InfographicVQA dataset to generate predictions for visual question answering tasks.

**For evaluation and scoring**, see [../eval/eval.md](../eval/eval.md).

## Data Format

Input JSONL files must contain:
```json
{
  "image": "37313.jpeg",
  "question": "Which social platform has heavy female audience?",
  "question_id": 98313,
  "answer": ["pinterest"],
  "answer_type": ["single span"],
  "evidence": ["text"],
  "operation/reasoning": []
}
```

**Note:** `answer` field accepts both string and list formats. All fields are preserved in output prediction files.

## Quick Start

### Basic Usage

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.inference.run_inference internvl
```

### Available Models

| Model ID | Model Name |
|----------|------------|
| `internvl` | InternVL3.5-8B |
| `qwenvl` | Qwen2.5-VL-7B |
| `minicpm` | MiniCPM-o-2-6 |
| `molmo` | Molmo-7B |
| `phi` | Phi-4-multimodal |
| `ovis` | Ovis2.5-9B |
| `videollama` | VideoLLAMA3-7B |

### Custom Arguments

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.inference.run_inference internvl \
    --image_folder /path/to/images \
    --data_path /path/to/data.jsonl \
    --output_dir ./results \
    --seed 42
```

**Default paths:**
- Images: `/mnt/VLAI_data/InfographicVQA/images`
- Data: `/mnt/VLAI_data/InfographicVQA/infographicvqa_val.jsonl`
- Output: `./results/`

## Output Format

Predictions are saved as JSON files in `results/` directory:

```json
[
  {
    "image": "37313.jpeg",
    "question": "Which social platform has heavy female audience?",
    "answer": ["pinterest"],
    "predict": "Pinterest",
    "answer_type": ["single span"],
    "evidence": ["text"],
    "operation/reasoning": []
  }
]
```

**Note:** Fields `anls`, `accuracy`, and `llm_score` are added during evaluation (see [eval.md](../eval/eval.md)).

## Project Structure

```
src/inference/
├── models/
│   ├── base_model.py      # Abstract base class
│   ├── utils.py           # get_prompt(), extract_clean_model_name()
│   ├── minicpm.py         # MiniCPM implementation
│   ├── molmo.py           # Molmo implementation
│   ├── internvl.py        # InternVL implementation
│   ├── ovis.py            # Ovis implementation
│   ├── phi.py             # Phi implementation
│   ├── qwenvl.py          # QwenVL implementation
│   ├── llava.py           # LLaVA implementation
│   └── videollama.py      # VideoLLAMA implementation
├── utils.py               # set_seed()
└── run_inference.py       # Main inference script

src/prompts/
└── vqa.jinja              # VQA prompt template
```

## Next Steps

After generating predictions, evaluate results using the scoring script. See [../eval/eval.md](../eval/eval.md) for details on computing Accuracy, ANLS, and LLM scores.

