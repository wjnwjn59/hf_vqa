CUDA_VISIBLE_DEVICES=1 torchrun -m llamafactory.cli \
    train \
    examples/train_lora/intern3_5vl_lora_sft.yaml