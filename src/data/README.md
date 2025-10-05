## Create BizGen Data

### Symbolic Links the Checkpoints

```bash
ln -s /mnt/VLAI_data/BizGen/checkpoints /home/thinhnp/hf_vqa/src/data/create_data/bizgen/
```

### Activate the environment

```bash
conda activate khoina_bizgen
``` 

### Run the script to create the data

```bash
cd /home/thinhnp/hf_vqa/src/data/create_data/bizgen/

python inference.py \
--ckpt_dir checkpoints/lora/infographic \
--output_dir infographic \
--sample_list /home/thinhnp/hf_vqa/src/data/create_data/output/bizgen_format/wiki000001.json
```

## Download the Wiki Data

```bash
cd /home/thinhnp/hf_vqa/src/data/create_data/wikipedia/

pip install -r requirements.txt

python download_wiki.py --output_dir wiki
```

## Process the Wiki Data

```bash
cd /home/thinhnp/hf_vqa/src/data/create_data/wikipedia/

python extract_wikipedia_intro.py \
    --dataset_path ./wikipedia_en_20231101 \
    --output_path ./wikipedia_processed \
    --min_words 1024 \
    --max_samples 232000 \
    --save_format json
```

## Process the Bboxes

```bash
cd /home/thinhnp/hf_vqa/src/data/create_data/qwen/

python extract_bboxes.py
```

## Create the Infographic Data

```bash
cd /home/thinhnp/hf_vqa/src/data/create_data/qwen/

python generate_infographic_data.py \
    --model_name "unsloth/Qwen3-8B" \
    --input_data "/home/thinhnp/hf_vqa/src/data/create_data/wikipedia/wikipedia_processed.json" \
    --template_path "/home/thinhnp/hf_vqa/src/prompts/bizgen.jinja" \
    --output_path "/home/thinhnp/hf_vqa/src/data/create_data/qwen/infographic_generated.json" \
    --batch_size 8 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_tokens 8192 \
    --gpu_memory_utilization 0.9 \
    --max_samples 8
```
