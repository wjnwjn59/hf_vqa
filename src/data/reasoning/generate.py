import os
import torch
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


MODEL_NAME = "/mnt/dataset1/pretrained_fm/Qwen_Qwen3-VL-8B-Thinking"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading model '{MODEL_NAME}' on {DEVICE}...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("✅ Model and processor loaded successfully.")

PROMPT_TEMPLATE = """
You are an expert AI assistant specializing in infographic analysis and visual reasoning.

Your task is to answer the question concisely based solely on the image.
Treat the following text as the complete representation of the infographic.

Infographic Content:
{layout_json_string}

Question: {question}

Answer:
""".strip()


def generate_answer_and_reasoning(
    image_path: str,
    layout_data: Dict[str, Any], # Nhận 1 object layout
    question: str
) -> str:
    
    # Chuyển object layout thành chuỗi JSON
    layout_json_string = json.dumps(layout_data, indent=2)
    
    # Định dạng prompt
    prompt = PROMPT_TEMPLATE.format(
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

    # Chuẩn bị inputs
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

    # Chạy inference
    generated_ids = model.generate(**inputs, **generation_params)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()

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
    # Lặp qua tất cả file wiki*.json
    for layout_file in tqdm(list(layout_dir.glob("wiki*.json")), desc="Loading Layouts"):
        # Lấy ID, ví dụ: "000001" từ "wiki000001.json"
        wiki_id = layout_file.stem.replace("wiki", "") 
        try:
            with open(layout_file, 'r', encoding='utf-8') as f:
                layouts = json.load(f) # Đây là một list
                # Tạo một dict map từ index (ví dụ: 1, 2) sang object layout tương ứng
                layout_data[wiki_id] = {item['index']: item for item in layouts}
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {layout_file}")
    print(f"✅ Loaded layout data for {len(layout_data)} wikis.")
    return layout_data

if __name__ == '__main__':
    SQUAD_FILE_PATH = Path("/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl")
    
    LAYOUT_DIR = Path("/home/thinhnp/hf_vqa/src/data/create_data/output/narrator_format_v2")
    LINK_DIR = Path("/home/thinhnp/hf_vqa/src/data/create_data/output/infographic_v2")
    IMAGE_DIR = Path("/home/thinhnp/hf_vqa/src/data/bizgen/output/squad_v2")
    OUTPUT_FILE_PATH = Path("./generated_results.jsonl")

    squad_data_map = load_squad_data(SQUAD_FILE_PATH)
    layout_data_map = load_layout_data(LAYOUT_DIR)

    print("\n" + "="*80)
    print("🚀 Starting generation process...")
    print(f"💾 Saving results to: {OUTPUT_FILE_PATH}")
    print("="*80)
    
    link_files = sorted(list(LINK_DIR.glob("infographic*.json")))
    
    with open(OUTPUT_FILE_PATH, 'a', encoding='utf-8') as f_out:
        
        # Vòng lặp chính: Lặp qua các file liên kết (ví dụ: infographic000001.json)
        for link_file in tqdm(link_files, desc="Processing Wikis", position=0):
            # Lấy wiki_id, ví dụ: "000001"
            wiki_id = link_file.stem.replace("infographic", "")
            
            try:
                with open(link_file, 'r', encoding='utf-8') as f:
                    link_list = json.load(f) # Đây là list các item liên kết
            except json.JSONDecodeError:
                tqdm.write(f"Skipping {link_file}: Could not parse JSON.")
                continue

            # Kiểm tra xem có layout data cho wiki_id này không
            if wiki_id not in layout_data_map:
                tqdm.write(f"Skipping Wiki {wiki_id}: No layout data found.")
                continue

            # Lặp qua từng câu hỏi trong file liên kết
            for link_item in tqdm(link_list, desc=f"Wiki {wiki_id}", position=1, leave=False):
                squad_id = link_item.get('id')
                layout_index = link_item.get('infographic_id') # Đây là index (ví dụ: 1, 2, 10...)

                # 1. Lấy dữ liệu từ các map đã tải trước
                qa_pair = squad_data_map.get(squad_id)
                sample_layout = layout_data_map.get(wiki_id, {}).get(layout_index)

                # Kiểm tra dữ liệu
                if not qa_pair:
                    tqdm.write(f"Skipping: Missing SQuAD data for squad_id {squad_id}")
                    continue
                if not sample_layout:
                    tqdm.write(f"Skipping: Missing layout for wiki {wiki_id}, index {layout_index}")
                    continue
                
                question = qa_pair["question"]
                ground_truth_answer = qa_pair["answer"]

                # 2. Xây dựng đường dẫn ảnh
                image_file_path = IMAGE_DIR / f"narrator{wiki_id}" / f"{layout_index}.png"

                if not image_file_path.exists():
                    tqdm.write(f"Skipping: Image not found at {image_file_path}")
                    continue

                tqdm.write("\n" + "="*50)
                tqdm.write(f"Processing: Wiki: {wiki_id}, Index: {layout_index}")
                tqdm.write(f"❓ Question: {question}")

                generated_output = generate_answer_and_reasoning(
                    image_path=str(image_file_path),
                    layout_data=sample_layout, # Truyền vào object layout cụ thể
                    question=question
                )

                # --- SỬA ĐỔI: Xử lý split an toàn hơn ---
                thinking = ""
                answer = ""
                if "</think>" in generated_output:
                    parts = generated_output.split("</think>", 1) # Split tối đa 1 lần
                    thinking = parts[0].strip()
                    # Xóa thẻ <think> ở đầu (nếu có)
                    if thinking.startswith("<think>"):
                        thinking = thinking[len("<think>"):].strip()
                    
                    answer = parts[1].strip()
                else:
                    answer = generated_output
                    tqdm.write("Warning: No '</think>' tag found in output.")

                tqdm.write(f"Thinking: {thinking}")
                tqdm.write(f"Answer: {answer}")
                tqdm.write(f"GT: {ground_truth_answer}")
                
                if ground_truth_answer.lower() in answer.lower():
                    tqdm.write("--- ✅ Match found! ---")
                else:
                    tqdm.write("--- ❌ No match. ---")
                tqdm.write("="*50)
                
                # --- BỔ SUNG: Tạo object kết quả và lưu vào file .jsonl ---
                result_item = {
                    "wiki_id": wiki_id,
                    "layout_index": layout_index,
                    "squad_id": squad_id,
                    "question": question,
                    "ground_truth_answer": ground_truth_answer,
                    "generated_full_output": generated_output,
                    "generated_thinking": thinking,
                    "generated_answer": answer,
                }
                
                # Ghi vào file jsonl (mỗi kết quả trên 1 dòng)
                f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                # --- KẾT THÚC BỔ SUNG ---

    print("\n🎉 All wikis processed successfully.")