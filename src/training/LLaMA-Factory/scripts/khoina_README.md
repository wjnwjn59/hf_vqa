# 1 Train
- Dùng file `scripts/train_internvl.sh`.
- Script train Internvl theo config trong file examples/train_lora/xyz.yaml.
- Khi chạy cần chỉnh sửa config đúng nhu cầu.
- Output sẽ là một folder (`save/`)chứa các checkpoints và các file config, thống kê, loss, ...

# 2 Merge LoRA
- Dùng lệnh: `llamafactory-cli export examples/merge_lora/intern3_5vl_lora_sft.yaml`.
- Lưu ý trỏ vào checkpoints phù hợp trong config.yaml.
- Output sau khi merge chứa trong folder `output/`.

# 3 Inference
- Dùng file `scripts/infer_intern.sh`.
- Script inference Internvl với các dataset.
- Output sẽ chứa trong folder `experiments/`.

# 4 Evaluate
- Dùng file `src/eval/calculate_scores.py`.
- Chỉnh lại path như sau `RESULTS_DIR = "training/LLaMA-Factory/experiments"`.
- Xem hướng dẫn chạy eval tại `src/eval/eval.md`.
- Output sẽ là các file kết quả trong thư mục `src/training/LLaMA-Factory/experiments/` và in ra ở terminal.

