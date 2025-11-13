CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun -m llamafactory.cli \
    train \
    examples/train_lora_qwen/qwen2vl_2b_instruct_lora_sft_generated_no_reasoning_10k.yaml