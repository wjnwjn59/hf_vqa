CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun -m llamafactory.cli \
    train \
    examples/train_lora_internvl/intern3_5vl_2b_lora_sft.yaml