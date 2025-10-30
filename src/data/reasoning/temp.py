import os
import torch
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader

# Cấu hình PyTorch CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Tải Model và Processor ---
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

# --- Tải Jinja Template ---
try:
    script_file_path = Path(__file__).resolve()
    src_dir = script_file_path.parent.parent.parent
except NameError:
    print("⚠️  '__file__' not defined. Assuming relative path for 'src'.")
    src_dir = Path.cwd().parent.parent 
    print(f"Set 'src_dir' to: {src_dir}")


template_dir = src_dir / "prompts"
template_name = "reasoning_generation.jinja"

try:
    jinja_env = Environment(loader=FileSystemLoader(template_dir))
    PROMPT_TEMPLATE_JINJA = jinja_env.get_template(template_name)
    print(f"✅ Template '{template_name}' loaded from '{template_dir}'.")
except Exception as e:
    print(f"❌ Error loading Jinja template '{template_name}' from '{template_dir}': {e}")
    print("Please make sure the file exists and paths are correct.")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Attempted 'src_dir': {src_dir}")
    exit(1)

# --- Hàm Sinh Lý Luận (Đã bỏ qua ảnh) ---
def generate_reasoning_chain(
    # image_path: str, # ĐÃ BỎ QUA
    layout_data: Dict[str, Any],
    question: str,
    ground_truth_answer: str
) -> str:
    """
    Sinh chuỗi lý luận (dưới dạng JSON) từ model Qwen3-VL.
    PHIÊN BẢN NÀY KHÔNG SỬ DỤNG ẢNH.
    """
    
    layout_json_string = json.dumps(layout_data, indent=2)
    
    prompt = PROMPT_TEMPLATE_JINJA.render(
        layout_json_string=layout_json_string,
        question=question,
        answer=ground_truth_answer
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                # {"type": "image", "image": image_path}, # ĐÃ BỎ QUA ẢNH
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
        "max_new_tokens": 4000,
    }

    generated_ids = model.generate(**inputs, **generation_params)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()

# --- Hàm Xử Lý Output JSON ---
def stitch_reasoning_json(json_string: str) -> str:
    """
    Phân tích JSON lý luận và thay thế các tham chiếu [ID] 
    để tạo ra một văn bản tự nhiên, liền mạch.
    (Giữ nguyên hàm này từ code cũ của bạn)
    """
    stitched_understand = "Error: Could not parse generated JSON."
    stitched_think = "Error: Could not parse generated JSON."
    stitched_answer = "Error: No answer found."
    data = {}
    try:
        data = json.loads(json_string.strip())
        replacement_map = {}

        # 2. Fill map from 'understand.relevant_elements'
        if "understand" in data and "relevant_elements" in data["understand"]:
            for el in data["understand"].get("relevant_elements", []):
                if 'id' in el:
                    key = f"[{el['id']}]"
                    desc = el.get('element_description', 'N/A')
                    coords = el.get('coordinates', 'N/A')
                    value = f"({desc} {coords})"
                    replacement_map[key] = value

        # 3. Fill map from 'think.evidence_array'
        if "think" in data and "evidence_array" in data["think"]:
            for ev in data["think"].get("evidence_array", []):
                if 'id' in ev:
                    key = f"[{ev['id']}]"
                    text = ev.get('text', 'N/A')
                    style = ev.get('text_style', 'N/A')
                    context = ev.get('spatial_context', 'N/A')
                    value = f"({text} [Style: {style}; Context: {context}])"
                    replacement_map[key] = value

        # 4. Xử lý văn bản 'understand.analysis'
        if "understand" in data and "analysis" in data["understand"]:
            stitched_understand = data["understand"]["analysis"]
            for key, value in replacement_map.items():
                stitched_understand = stitched_understand.replace(key, value)
        else:
            stitched_understand = "Error: Missing 'understand' or 'analysis' structure."

        # 5. Xử lý văn bản 'think.logical_reasoning'
        if "think" in data and "logical_reasoning" in data["think"]:
            stitched_think = data["think"]["logical_reasoning"]
            for key, value in replacement_map.items():
                stitched_think = stitched_think.replace(key, value)
        else:
            stitched_think = "Error: Missing 'think' or 'logical_reasoning' structure."

        # 6. Lấy câu trả lời
        stitched_answer = data.get('answer', 'Error: Missing "answer" key.')

    except json.JSONDecodeError:
        stitched_understand = f"Error: Failed to decode JSON. Raw output: {json_string[:200]}..."
        stitched_think = "Error: Failed to decode JSON."
        stitched_answer = "Error: Failed to decode JSON."
    except Exception as e:
        stitched_understand = f"Error during stitching: {e}"
        stitched_think = f"Error during stitching: {e}"
        stitched_answer = f"Error during stitching: {e}"

    # 7. Trả về một đoạn văn duy nhất, liền mạch.
    if not stitched_understand.strip().endswith('.'):
        stitched_understand += '.'
    
    if not stitched_think.strip().endswith('.'):
        stitched_think += '.'

    return (
        f"{stitched_understand} "
        f"{stitched_think} "
        f"Therefore, the answer is {stitched_answer}."
    )

# --- Hàm Main Thực Thi ---
if __name__ == '__main__':
    # --- Định nghĩa các đường dẫn ---
    # Đường dẫn thư mục chứa các file JSON mới (ví dụ: wiki000001.json)
    LAYOUT_DIR = Path("/home/binhdt/hf_vqa/src/data/reasoning/")
    
    # Đường dẫn thư mục chứa ảnh (ĐÃ BỎ QUA)
    # IMAGE_DIR = Path("/home/thinhnp/hf_vqa/src/data/bizgen/output/squad_v2") 
    
    # Đường dẫn file output
    OUTPUT_FILE_PATH = Path("./reasoning_results_no_image.jsonl")

    print("\n" + "="*80)
    print(f"💾 Saving results to: {OUTPUT_FILE_PATH}")
    
    # Lấy tất cả các file wiki JSON từ thư mục LAYOUT_DIR
    wiki_files = sorted(list(LAYOUT_DIR.glob("wiki*.json")))
    
    if not wiki_files:
        print(f"❌ No 'wiki*.json' files found in {LAYOUT_DIR}. Please check the path.")
        exit(1)

    print(f"Found {len(wiki_files)} wiki files to process.")

    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f_out:
        
        # Vòng lặp 1: Duyệt qua từng file wiki (ví dụ: wiki000001.json)
        for wiki_file in tqdm(wiki_files, desc="Processing Wiki Files", position=0):
            
            # Trích xuất wiki_id (ví dụ: "000001")
            wiki_id = wiki_file.stem.replace("wiki", "") 
            
            try:
                with open(wiki_file, 'r', encoding='utf-8') as f:
                    # Mỗi file là một DANH SÁCH các item (layout)
                    wiki_data_list = json.load(f)
            except json.JSONDecodeError:
                tqdm.write(f"Skipping {wiki_file}: Could not parse JSON.")
                continue

            # Vòng lặp 2: Duyệt qua từng item (layout) trong file
            for item in tqdm(wiki_data_list, desc=f"Wiki {wiki_id}", position=1, leave=False):
                
                layout_index = item.get('index')
                # Toàn bộ 'item' này chính là dữ liệu layout
                sample_layout = item 
                
                if layout_index is None: 
                    tqdm.write(f"Skipping item in {wiki_file}: Missing 'index'.")
                    continue

                # Xây dựng đường dẫn ảnh (ĐÃ BỎ QUA THEO YÊU CẦU)
                # image_file_path = IMAGE_DIR / f"narrator{wiki_id}" / f"{layout_index}.png"
                # if not image_file_path.exists():
                #     tqdm.write(f"Skipping: Image not found at {image_file_path}")
                #     continue
                
                # Lấy danh sách các cặp Q&A
                qa_pairs = item.get('original_qa_pairs', [])
                if not qa_pairs:
                    tqdm.write(f"Skipping item {layout_index} in {wiki_file}: No 'original_qa_pairs' found.")
                    continue

                # Vòng lặp 3: Duyệt qua từng cặp Q&A cho item này
                for qa_pair in qa_pairs:
                    question = qa_pair.get('question')
                    squad_id = qa_pair.get('id')
                    
                    # Lấy câu trả lời đầu tiên
                    answers_list = qa_pair.get('answers', {}).get('text', [])
                    
                    if not answers_list:
                        tqdm.write(f"Skipping QA {squad_id}: No answers text found.")
                        continue
                    ground_truth_answer = answers_list[0]

                    if not question or not squad_id:
                        tqdm.write(f"Skipping QA in {wiki_file} (index {layout_index}): Missing 'question' or 'id'.")
                        continue

                    tqdm.write("\n" + "="*50)
                    tqdm.write(f"Processing: Wiki: {wiki_id}, Index: {layout_index}, SQuAD ID: {squad_id}")
                    tqdm.write(f"❓ Question: {question}")
                    tqdm.write(f"🎯 Ground Truth (Provided): {ground_truth_answer}")
                    tqdm.write("⏳ Generating reasoning chain (NO IMAGE)...")
                    
                    # Gọi hàm sinh lý luận (không truyền ảnh)
                    generated_output = generate_reasoning_chain(
                        # image_path=str(image_file_path), # ĐÃ BỎ QUA
                        layout_data=sample_layout,
                        question=question,
                        ground_truth_answer=ground_truth_answer
                    )
                    
                    # --- Xử lý hậu kỳ (giữ nguyên) ---
                    generated_reasoning_output = ""
                    if "</think>" in generated_output:
                        parts = generated_output.split("</think>", 1)
                        generated_reasoning_output = parts[1].strip()
                    else:
                        generated_reasoning_output = generated_output.strip()
                    
                    json_start_index = generated_reasoning_output.find('{')
                    if json_start_index != -1:
                        generated_reasoning_output = generated_reasoning_output[json_start_index:]
                    
                    json_end_index = generated_reasoning_output.rfind('}')
                    if json_end_index != -1:
                        generated_reasoning_output = generated_reasoning_output[:json_end_index+1]
                    # --- Kết thúc xử lý hậu kỳ ---

                    tqdm.write("✅ Raw Reasoning JSON (Cleaned):\n")
                    tqdm.write(generated_reasoning_output)
                    
                    # Ghép nối JSON thành văn bản tự nhiên
                    stitched_reasoning = stitch_reasoning_json(generated_reasoning_output)
                    tqdm.write("\n🔍 Merged Reasoning:\n")
                    tqdm.write(stitched_reasoning)

                    # Ghi kết quả
                    result_item = {
                        "wiki_id": wiki_id,
                        "layout_index": layout_index,
                        "squad_id": squad_id,
                        "question": question,
                        "ground_truth_answer": ground_truth_answer,
                        "generated_reasoning": generated_reasoning_output,
                        "merged_reasoning": stitched_reasoning 
                    }
                    
                    f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')

    print("\n🎉 All wiki files processed successfully (without images).")