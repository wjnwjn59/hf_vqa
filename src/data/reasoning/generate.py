import os
import torch
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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
script_file_path = Path(__file__).resolve()
src_dir = script_file_path.parent.parent.parent

template_dir = src_dir / "prompts"
template_name = "reasoning_generation.jinja"

try:
    jinja_env = Environment(loader=FileSystemLoader(template_dir))
    PROMPT_TEMPLATE_JINJA = jinja_env.get_template(template_name)
    print(f"✅ Template '{template_name}' loaded from '{template_dir}'.")
except Exception as e:
    print(f"❌ Error loading Jinja template '{template_name}' from '{template_dir}': {e}")
    print("Please make sure the file exists and paths are correct.")
    exit(1)


def generate_reasoning_chain(
    image_path: str,
    layout_data: Dict[str, Any],
    question: str,
    ground_truth_answer: str
) -> str:
    
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
        "max_new_tokens": 4000,
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

def stitch_reasoning_json(json_string: str) -> str:
    """
    Analyze the reasoning JSON and replace [ID] references to create a natural, cohesive text.
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
        stitched_answer = data.get('answer', 'Error: Missing "answer" key.') # Sửa lỗi KeyError

    except json.JSONDecodeError:
        stitched_understand = f"Error: Failed to decode JSON. Raw output: {json_string[:200]}..."
        stitched_think = "Error: Failed to decode JSON."
        stitched_answer = "Error: Failed to decode JSON."
    except Exception as e:
        stitched_understand = f"Error during stitching: {e}"
        stitched_think = f"Error during stitching: {e}"
        stitched_answer = f"Error during stitching: {e}"

    # 7. Trả về một đoạn văn duy nhất, liền mạch.
    # Thêm dấu chấm nếu bị thiếu để đảm bảo ngữ pháp
    if not stitched_understand.strip().endswith('.'):
        stitched_understand += '.'
    
    if not stitched_think.strip().endswith('.'):
        stitched_think += '.'

    return (
        f"{stitched_understand} "
        f"{stitched_think} "
        f"Therefore, the answer is {stitched_answer}."
    )


if __name__ == '__main__':
    SQUAD_FILE_PATH = Path("/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl")
    # LAYOUT_DIR = Path("/home/thinhnp/hf_vqa/src/data/create_data/output/narrator_format_v2")
    LAYOUT_DIR = Path("/home/binhdt/hf_vqa/src/data/reasoning/")
    LINK_DIR = Path("/home/thinhnp/hf_vqa/src/data/create_data/output/infographic_v2")
    IMAGE_DIR = Path("/home/thinhnp/hf_vqa/src/data/bizgen/output/squad_v2")
    OUTPUT_FILE_PATH = Path("./50_results.jsonl")

    squad_data_map = load_squad_data(SQUAD_FILE_PATH)
    layout_data_map = load_layout_data(LAYOUT_DIR)

    print("\n" + "="*80)
    print(f"💾 Saving results to: {OUTPUT_FILE_PATH}")
    
    link_files = sorted(list(LINK_DIR.glob("infographic*.json")))
    
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f_out:
        
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
                tqdm.write(f"🎯 Ground Truth (Provided): {ground_truth_answer}")
                tqdm.write("⏳ Generating reasoning chain...")
                
                generated_output = generate_reasoning_chain(
                    image_path=str(image_file_path),
                    layout_data=sample_layout,
                    question=question,
                    ground_truth_answer=ground_truth_answer
                )
                
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


                tqdm.write("✅ Raw Reasoning JSON (Cleaned):\n")
                tqdm.write(generated_reasoning_output)
                
                stitched_reasoning = stitch_reasoning_json(generated_reasoning_output)
                tqdm.write("\n🔍 Merged Reasoning:\n")
                tqdm.write(stitched_reasoning)

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

    print("\n🎉 All wikis processed successfully.")