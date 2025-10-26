# SFT-Kit (Basic English)

- Clone and install LLaMA-Factory
  - `git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && cd LLaMA-Factory`
  - `pip install -e ".[torch,metrics]" --no-build-isolation`
  - If something is missing: `pip install -r requirements.txt`

- Prepare data
  - Copy `conversation_squad_v2_train.jsonl` into `data/`
  - Replace default `dataset_info.json` with the project one

- Train SFT (LoRA)
  - Put configs into `examples/train_lora/`: `intern3_5vl_lora_sft.yaml`, `qwen2_5vl_lora_sft.yaml`
  - Check and edit paths and hyperparameters in YAML
  - Run: `CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m llamafactory.cli train examples/train_lora/intern3_5vl_lora_sft.yaml`
  - Output: a checkpoints folder (used for merge)

- Merge LoRA
  - Use config in `examples/merge_lora/`: `intern3_5vl_lora_sft.yaml` (similar for Qwen)
  - Set `adapter_name_or_path` to the latest checkpoint
  - Export merged weights: `CUDA_VISIBLE_DEVICES=0 llamafactory-cli export examples/merge_lora/intern3_5vl_lora_sft.yaml`
  - Output: merged weights + config

- Inference
  - cd to `hf_vqa`; update `internvl_lora.py` if needed
  - Run: `CUDA_VISIBLE_DEVICES=0 python -m src.inference.run_inference internvl`
  - JSON results: `src/inference/results/<ModelName>_<Dataset>.json`

- Evaluation
  - cd to `hf_vqa/src/eval`
  - Quick evaluate (no LLM score): `CUDA_VISIBLE_DEVICES=0 python -m src.eval.calculate_scores --no-llm-scores`
