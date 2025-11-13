# dataset_name=$1
# mode=$2

# if [ -z "$dataset_name" ] || [ -z "$mode" ]; then
#     echo "Usage: $0 <dataset_name> <mode>"
#     exit 1
# fi

# case $dataset_name in
#     infographicvqa)
#         dataset="infographicvqa_val"
#         json_template_path="data/infographicvqa_val_lmf.jsonl"
#         ;;
#     infographicvqa_test)
#         dataset="infographicvqa_test"
#         json_template_path="data/infographicvqa_test_lmf.jsonl"
#         ;;
#     docvqa)
#         dataset="docvqa_val"
#         json_template_path="data/docvqa_val_lmf.jsonl"
#         ;;
#     textvqa)
#         dataset="textvqa_val"
#         json_template_path="data/textvqa_val_lmf.jsonl"
#         ;;
#     *)
#         echo "Unknown dataset: $dataset_name"
#         exit 1
#         ;;
# esac

# case $mode in
#     base)
#         model_name_or_path="OpenGVLab/InternVL3_5-2B-HF"
#         json_save_name="experiments/intern3_5_${dataset_name}_base.json"
#         ;;
#     sft)
#         model_name_or_path="output/intern3_5vl_2b_sft"
#         json_save_name="experiments/intern3_5_${dataset_name}_sft.json"
#         ;;
#     sft_reasoning)
#         model_name_or_path="output/intern3_5vl_2b_sft_reasoning"
#         json_save_name="experiments/intern3_5_${dataset_name}_sft_reasoning.json"
#         ;;
#     sft_vqaprompt)
#         model_name_or_path="output/intern3_5vl_2b_sft_vqaprompt"
#         json_save_name="experiments/intern3_5_${dataset_name}_sft_vqaprompt.json"
#         ;;
#     sft_reasoning_vqaprompt)
#         model_name_or_path="output/intern3_5vl_2b_sft_reasoning_vqaprompt"
#         json_save_name="experiments/intern3_5_${dataset_name}_sft_reasoning_vqaprompt.json"
#         ;;
#     *)
#         echo "Unknown model_name_or_path: $model_name_or_path"
#         exit 1
#         ;;
# esac


CUDA_VISIBLE_DEVICES=0 python3 scripts/vllm_infer_with_json.py \
    --model_name_or_path output/qwen3vl_instruct_2b_sft_narrative_origin_only_no_reasoning \
    --dataset infographicvqa_test \
    --template qwen3_vl \
    --batch_size 2048 \
    --default_system "" \
    --cutoff_len 4096 \
    --max_new_tokens 128 \
    --seed 42 \
    --vllm_config '{"gpu_memory_utilization": 0.9, "max_model_len": 4224, "max_num_batched_tokens": 12672}' \
    --json_template_path data/infographicvqa_test_lmf.jsonl \
    --json_save_name experiments/qwen3vl.json \
    --temperature 0.0 \
    --image_max_pixels 2408448 \
    --enable_thinking false
