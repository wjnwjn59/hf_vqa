import os
import torch
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
script_file_path = Path(__file__).resolve()
src_dir = script_file_path.parent.parent.parent

template_dir = src_dir / "prompts"
template_name = "qa_generation.jinja"

try:
    jinja_env = Environment(loader=FileSystemLoader(template_dir))
    PROMPT_TEMPLATE_JINJA = jinja_env.get_template(template_name)
    print(f"✅ Template '{template_name}' loaded from '{template_dir}'.")
except Exception as e:
    print(f"❌ Error loading Jinja template '{template_name}' from '{template_dir}': {e}")
    print("Please make sure the file exists and paths are correct.")
    exit(1)
    

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


def generate_new_qas(
    image_path: str,
    layout_data: Dict[str, Any],
    qa_samples: List[Dict[str, Any]], 
    k: int                             
) -> str:
    
    layout_json_string = json.dumps(layout_data, indent=2)
    
    prompt = PROMPT_TEMPLATE_JINJA.render(
        layout_json_string=layout_json_string,
        qa_samples=qa_samples,
        k=k
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
        "max_new_tokens": 5000, 
    }

    generated_ids = model.generate(**inputs, **generation_params)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()

def load_squad_data(filepath: Path) -> Dict[str, Dict[str, Any]]:
    squad_data = {}
    print(f"Loading SQuAD data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get('answers', {}).get('text'):
                squad_data[item['id']] = {
                    "question": item['question'],
                    "answer": item['answers']['text'][0]
                }
    print(f"✅ Loaded {len(squad_data)} SQuAD items.")
    return squad_data

def load_layout_data(layout_dir: Path) -> Dict[str, Dict[int, Any]]:
    layout_data = {}
    print(f"Loading layout data from {layout_dir}...")
    for layout_file in tqdm(list(layout_dir.glob("wiki*.json")), desc="Loading Layouts"):
        wiki_id = layout_file.stem.replace("wiki", "") 
        try:
            with open(layout_file, 'r', encoding='utf-8') as f:
                layouts = json.load(f)
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
    OUTPUT_FILE_PATH = Path("./generated_qa.jsonl") 
    # Số lượng K QA mới cần tạo cho mỗi layout
    K_VALUE = 3 

    # --- Load dữ liệu cơ sở ---
    squad_data_map = load_squad_data(SQUAD_FILE_PATH)
    layout_data_map = load_layout_data(LAYOUT_DIR)

    # --- Bước tiền xử lý: Nhóm các QA theo (wiki_id, layout_index) ---
    print("Building layout-to-QA map...")
    layout_to_qas_map = {}
    link_files = sorted(list(LINK_DIR.glob("infographic*.json")))
    
    for link_file in tqdm(link_files, desc="Mapping QAs to Layouts"):
        wiki_id = link_file.stem.replace("infographic", "")
        try:
            with open(link_file, 'r', encoding='utf-8') as f:
                link_list = json.load(f)
        except json.JSONDecodeError:
            tqdm.write(f"Skipping {link_file}: Could not parse JSON.")
            continue

        for link_item in link_list:
            squad_id = link_item.get('id')
            layout_index = link_item.get('infographic_id')
            qa_pair = squad_data_map.get(squad_id)
            
            if not qa_pair:
                continue
                
            map_key = (wiki_id, layout_index)
            if map_key not in layout_to_qas_map:
                layout_to_qas_map[map_key] = []
                
            # Thêm ID cho prompt (Q1, A1, Q2, A2...)
            qa_with_id = {
                "id": len(layout_to_qas_map[map_key]) + 1,
                "question": qa_pair["question"],
                "answer": qa_pair["answer"]
            }
            layout_to_qas_map[map_key].append(qa_with_id)
    
    print(f"✅ Built map for {len(layout_to_qas_map)} unique layouts.")
    
    print("\n" + "="*80)
    print(f"💾 Saving new QAs to: {OUTPUT_FILE_PATH}")
    
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f_out:
        
        # Lặp qua map đã nhóm
        for (wiki_id, layout_index), qa_samples in tqdm(layout_to_qas_map.items(), desc="Generating New QAs"):
            
            sample_layout = layout_data_map.get(wiki_id, {}).get(layout_index)
            if not sample_layout:
                tqdm.write(f"Skipping ({wiki_id}, {layout_index}): Layout data missing.")
                continue
            
            image_file_path = IMAGE_DIR / f"narrator{wiki_id}" / f"{layout_index}.png"
            if not image_file_path.exists():
                tqdm.write(f"Skipping: Image not found at {image_file_path}")
                continue

            tqdm.write("\n" + "="*50)
            tqdm.write(f"Processing: Wiki: {wiki_id}, Index: {layout_index}")
            tqdm.write(f"Found {len(qa_samples)} existing QAs.")
            tqdm.write(f"⏳ Generating {K_VALUE} new QAs...")
            
            generated_output = generate_new_qas(
                image_path=str(image_file_path),
                layout_data=sample_layout,
                qa_samples=qa_samples, # Truyền danh sách QA đã có
                k=K_VALUE
            )
            
            # --- Xử lý output (chỉ lấy JSON array) ---
            generated_json_string = ""
            if "</think>" in generated_output:
                parts = generated_output.split("</think>", 1)
                generated_json_string = parts[1].strip()
            else:
                generated_json_string = generated_output.strip()

            # Tìm JSON array `[` ... `]`
            json_start_index = generated_json_string.find('[')
            json_end_index = generated_json_string.rfind(']')
            
            if json_start_index != -1 and json_end_index != -1:
                generated_json_string = generated_json_string[json_start_index : json_end_index + 1]
            else:
                generated_json_string = "[]" # Mặc định là array rỗng nếu không tìm thấy

            tqdm.write("✅ Raw Generated JSON Array (Cleaned):\n")
            tqdm.write(generated_json_string)

            result_item = {
                "wiki_id": wiki_id,
                "layout_index": layout_index,
                "existing_qas_count": len(qa_samples),
                "generated_qas_raw": generated_json_string,
                "k_requested": K_VALUE
            }
            
            f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')

    print("\n🎉 All new QAs generated successfully.")