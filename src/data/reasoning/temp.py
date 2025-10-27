import os
import torch
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm

# Đặt biến môi trường
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 1. Tải Model ---

MODEL_NAME = "/mnt/dataset1/pretrained_fm/Qwen_Qwen3-VL-8B-Thinking"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading model '{MODEL_NAME}' on {DEVICE}...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("✅ Model and processor loaded successfully.")

# --- 2. Định nghĩa 2 PROMPT TEMPLATES (Giữ nguyên) ---

# Mode 1: Model tự sinh ra Answer + Reasoning (dưới dạng <think>)
PROMPT_GENERATE_ALL = """
You are an expert AI assistant specializing in infographic analysis and visual reasoning.
Your task is to answer the question concisely based solely on the image.
Treat the following text as the complete representation of the infographic.

Infographic Content:
{layout_json_string}

Question: {question}

Answer:
""".strip()

# Mode 2: Cung cấp Answer, Model chỉ sinh Reasoning
PROMPT_GENERATE_REASONING = """
You are an expert AI assistant specializing in infographic analysis and visual reasoning.
Your task is to generate a concise reasoning chain that explains how to find the final answer using only the provided infographic image.
Treat the following text as the complete representation of the infographic.

Infographic Content:
{layout_json_string}

Question: {question}

Ground-Truth Answer: {answer}

Reasoning:
""".strip()

# --- 3. Hàm Inference (Giữ nguyên) ---

def run_model_inference(
    image_path: str,
    layout_data: Dict[str, Any],
    question: str,
    ground_truth_answer: str = None  # Tham số tùy chọn để chọn Mode
) -> str:
    
    layout_json_string = json.dumps(layout_data, indent=2)
    
    # --- Logic chọn Prompt ---
    if ground_truth_answer:
        # Mode 2: Cung cấp câu trả lời, yêu cầu lý do
        prompt = PROMPT_GENERATE_REASONING.format(
            layout_json_string=layout_json_string,
            question=question,
            answer=ground_truth_answer
        )
    else:
        # Mode 1: Yêu cầu cả câu trả lời và lý do (ẩn)
        prompt = PROMPT_GENERATE_ALL.format(
            layout_json_string=layout_json_string,
            question=question
        )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    generation_params = {
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    generated_ids = model.generate(**inputs, **generation_params)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()

# --- 4. Các hàm tải dữ liệu (Giữ nguyên) ---

def load_squad_data(filepath: Path) -> Dict[str, Dict[str, Any]]:
    """Tải file SQuAD JSONL vào một dictionary để tra cứu nhanh."""
    squad_data = {}
    print(f"Loading SQuAD data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get('answers', {}).get('text'):
                squad_data[item['id']] = {
                    "question": item['question'],
                    "answer": item['answers']['text'][0] # Lấy câu trả lời đầu tiên
                }
    print(f"✅ Loaded {len(squad_data)} SQuAD items.")
    return squad_data

def load_layout_data(layout_dir: Path) -> Dict[str, Dict[int, Any]]:
    """Tải tất cả layout vào một dict lồng: {wiki_id: {layout_index: layout_obj}}."""
    layout_data = {}
    print(f"Loading layout data from {layout_dir}...")
    for layout_file in tqdm(list(layout_dir.glob("wiki*.json")), desc="Loading Layouts"):
        wiki_id = layout_file.stem.replace("wiki", "") 
        try:
            with open(layout_file, 'r', encoding='utf-8') as f:
                layouts = json.load(f) # Đây là một list
                layout_data[wiki_id] = {item['index']: item for item in layouts}
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {layout_file}")
    print(f"✅ Loaded layout data for {len(layout_data)} wikis.")
    return layout_data


if __name__ == '__main__':
    # --- !!! CONTROL FLAG: CHỌN CHẾ ĐỘ CHẠY TẠI ĐÂY !!! ---
    # True: Chạy Mode 2 (cung cấp Answer, model chỉ sinh Reasoning)
    # False: Chạy Mode 1 (model tự sinh Answer + Reasoning)
    RUN_MODE_GENERATE_REASONING = True 
    # ------------------------------------------------------

    # --- Định nghĩa các đường dẫn ---
    SQUAD_FILE_PATH = Path("/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl")
    LAYOUT_DIR = Path("/home/thinhnp/hf_vqa/src/data/create_data/output/narrator_format_v2")
    LINK_DIR = Path("/home/thinhnp/hf_vqa/src/data/create_data/output/infographic_v2")
    IMAGE_DIR = Path("/home/thinhnp/hf_vqa/src/data/bizgen/output/squad_v2")
    OUTPUT_FILE_PATH = Path(f"./generated_results_{'reasoning_only' if RUN_MODE_GENERATE_REASONING else 'generate_all'}.jsonl")

    # --- Tải trước dữ liệu ---
    squad_data_map = load_squad_data(SQUAD_FILE_PATH)
    layout_data_map = load_layout_data(LAYOUT_DIR)

    print("\n" + "="*80)
    print(f"🚀 Starting generation process (Mode: {'Reasoning Only' if RUN_MODE_GENERATE_REASONING else 'Generate All'})")
    print(f"💾 Saving results to: {OUTPUT_FILE_PATH}")
    print("="*80)
    
    link_files = sorted(list(LINK_DIR.glob("infographic*.json")))
    
    with open(OUTPUT_FILE_PATH, 'a', encoding='utf-8') as f_out:
        
        for link_file in tqdm(link_files, desc="Processing Wikis", position=0):
            wiki_id = link_file.stem.replace("infographic", "")
            
            try:
                with open(link_file, 'r', encoding='utf-8') as f:
                    link_list = json.load(f)
            except json.JSONDecodeError:
                tqdm.write(f"Skipping {link_file}: Could not parse JSON.")
                continue

            if wiki_id not in layout_data_map:
                tqdm.write(f"Skipping Wiki {wiki_id}: No layout data found.")
                continue

            for link_item in tqdm(link_list, desc=f"Wiki {wiki_id}", position=1, leave=False):
                squad_id = link_item.get('id')
                layout_index = link_item.get('infographic_id')

                qa_pair = squad_data_map.get(squad_id)
                sample_layout = layout_data_map.get(wiki_id, {}).get(layout_index)

                if not qa_pair or not sample_layout:
                    continue
                
                question = qa_pair["question"]
                ground_truth_answer = qa_pair["answer"]
                image_file_path = IMAGE_DIR / f"narrator{wiki_id}" / f"{layout_index}.png"

                if not image_file_path.exists():
                    tqdm.write(f"Skipping: Image not found at {image_file_path}")
                    continue

                tqdm.write("\n" + "="*50)
                tqdm.write(f"Processing: Wiki: {wiki_id}, Index: {layout_index}")
                tqdm.write(f"❓ Question: {question}")
                
                result_item = {} # Chuẩn bị object để lưu

                if RUN_MODE_GENERATE_REASONING:
                    # --- MODE 2: GENERATE REASONING ---
                    tqdm.write(f"🎯 Ground Truth (Provided): {ground_truth_answer}")
                    tqdm.write("⏳ Generating reasoning chain (Mode 2)...")
                    
                    generated_output = run_model_inference(
                        image_path=str(image_file_path),
                        layout_data=sample_layout,
                        question=question,
                        ground_truth_answer=ground_truth_answer # Cung cấp GT
                    )
                    
                    # --- Logic split <think> cho Mode 2 ---
                    generated_reasoning_output = "" # Phần sau tag </think>
                    if "</think>" in generated_output:
                        parts = generated_output.split("</think>", 1)
                        generated_reasoning_output = parts[1].strip()
                    else:
                        generated_reasoning_output = generated_output # Toàn bộ là output

                    # Xóa tiền tố "Reasoning:" nếu có
                    if generated_reasoning_output.startswith("Reasoning:"):
                        generated_reasoning_output = generated_reasoning_output[len("Reasoning:"):].strip()

                    tqdm.write("✅ Full Output:\n")
                    tqdm.write(generated_output)
                    tqdm.write("✅ Reasoning Parsed:\n")
                    tqdm.write(generated_reasoning_output)

                    result_item = {
                        "mode": "generate_reasoning",
                        "wiki_id": wiki_id,
                        "layout_index": layout_index,
                        "squad_id": squad_id,
                        "question": question,
                        "ground_truth_answer": ground_truth_answer,
                        "generated_reasoning": generated_reasoning_output,
                        "generated_answer": None # Model không sinh answer
                    }

                else:
                    # --- MODE 1: GENERATE ALL ---
                    tqdm.write(f"🎯 Ground Truth (Hidden): {ground_truth_answer}")
                    tqdm.write("⏳ Generating answer and reasoning (Mode 1)...")
                    
                    generated_output = run_model_inference(
                        image_path=str(image_file_path),
                        layout_data=sample_layout,
                        question=question,
                        ground_truth_answer=None # KHÔNG cung cấp GT
                    )
                    
                    # --- THAY ĐỔI: Logic split <think> cho Mode 1 ---
                    generated_reasoning = "" # Phần trước </think>
                    generated_answer = ""    # Phần sau </think>
                    if "</think>" in generated_output:
                        parts = generated_output.split("</think>", 1)
                        generated_reasoning = parts[0].strip()
                        if generated_reasoning.startswith("<think>"):
                            generated_reasoning = generated_reasoning[len("<think>"):].strip()
                        generated_answer = parts[1].strip()
                    else:
                        generated_reasoning = "" # Không có <think>
                        generated_answer = generated_output # Toàn bộ là answer

                    # Xóa tiền tố "Answer:" nếu có
                    if generated_answer.startswith("Answer:"):
                        generated_answer = generated_answer[len("Answer:"):].strip()

                    tqdm.write(f"✅ Generated Reasoning (Thinking): {generated_reasoning}")
                    tqdm.write(f"✅ Generated Answer: {generated_answer}")
                    
                    if ground_truth_answer.lower() in generated_answer.lower():
                        tqdm.write("--- ✅ Match found! ---")
                    else:
                        tqdm.write("--- ❌ No match. ---")

                    result_item = {
                        "mode": "generate_all",
                        "wiki_id": wiki_id,
                        "layout_index": layout_index,
                        "squad_id": squad_id,
                        "question": question,
                        "ground_truth_answer": ground_truth_answer,
                        "generated_reasoning": generated_reasoning, # Gán phần thinking vào reasoning
                        "generated_answer": generated_answer
                    }
                    # --- KẾT THÚC THAY ĐỔI ---

                # Ghi kết quả vào file .jsonl
                f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                tqdm.write("="*50)

    print("\n🎉 All wikis processed successfully.")