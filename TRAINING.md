## Convert image and annotions to training format

```bash
export PYTHONPATH="./:$PYTHONPATH"
conda activate wiki

python src/data/narrator/convert_to_training_format.py \
    --qa-file /mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl \
    --image-base-dir src/data/bizgen/output \
    --dataset-name squad_v2 \
    --dataset-type squad_v2 \
    --output-file src/data/narrator/output/conversation_squad_v2_train.json
```

## Run training script

```bash
conda env create -f ./src/training/Qwen2-VL-Finetune/environment.yaml
conda activate qwen2_train

cd src/training/Qwen2-VL-Finetune
bash ./scripts/finetune_narrator.sh
```