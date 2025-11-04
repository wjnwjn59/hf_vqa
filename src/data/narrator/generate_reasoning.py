import os
import torch
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


MODEL_NAME = "/mnt/dataset1/pretrained_fm/Qwen_Qwen3-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading model '{MODEL_NAME}' on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("✅ Model and tokenizer loaded successfully.") 
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

def generate_reasoning_chain(
    layout_data: Dict[str, Any],
    question: str,
    ground_truth_answer: str
) -> (str, str): 
    
    layout_json_string = json.dumps(layout_data, indent=2)
    
    prompt = PROMPT_TEMPLATE_JINJA.render(
        layout_json_string=layout_json_string,
        question=question,
        answer=ground_truth_answer
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,   tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=5000
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0 

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return thinking_content.strip(), content.strip()

def stitch_reasoning_json(data: Dict[str, Any]) -> str:
    stitched_understand = "Error: Could not parse dictionary."
    stitched_think = "Error: Could not parse dictionary."
    stitched_answer = "Error: No answer found."
    try:
        replacement_map = {}

        if "understand" in data and "relevant_elements" in data["understand"]:
            for el in data["understand"].get("relevant_elements", []):
                if 'id' in el:
                    key = f"[{el['id']}]"
                    desc = el.get('element_description', 'N/A')
                    coords = el.get('coordinates', 'N/A')
                    value = f"({desc} {coords})"
                    replacement_map[key] = value

        if "think" in data and "evidence_array" in data["think"]:
            for ev in data["think"].get("evidence_array", []):
                if 'id' in ev:
                    key = f"[{ev['id']}]"
                    text = ev.get('text', 'N/A')
                    style = ev.get('text_style', 'N/A')
                    context = ev.get('spatial_context', 'N/A')
                    value = f"({text} [Style: {style}; Context: {context}])"
                    replacement_map[key] = value

        if "understand" in data and "analysis" in data["understand"]:
            stitched_understand = data["understand"]["analysis"]
            for key, value in replacement_map.items():
                stitched_understand = stitched_understand.replace(key, value)
        else:
            stitched_understand = "Error: Missing 'understand' or 'analysis' structure."

        if "think" in data and "logical_reasoning" in data["think"]:
            stitched_think = data["think"]["logical_reasoning"]
            for key, value in replacement_map.items():
                stitched_think = stitched_think.replace(key, value)
        else:
            stitched_think = "Error: Missing 'think' or 'logical_reasoning' structure."

        stitched_answer = data.get('answer', 'Error: Missing "answer" key.')

    except Exception as e:
        stitched_understand = f"Error during stitching: {e}"
        stitched_think = f"Error during stitching: {e}"
        stitched_answer = f"Error during stitching: {e}"

    return (
        f"{stitched_understand} "
        f"{stitched_think} "
        f"Therefore, the answer is {stitched_answer}."
    )

if __name__ == '__main__':
    LAYOUT_DIR = Path("/home/binhdt/hf_vqa/src/data/wiki/")
    OUTPUT_FILE_PATH = Path("../narrator/generated_reasonings.jsonl")
    
    print("\n" + "="*80)
    print(f"💾 Saving results to: {OUTPUT_FILE_PATH} (appending)")
    
    wiki_files = sorted(list(LAYOUT_DIR.glob("wiki*.json")))
    
    if not wiki_files:
        print(f"❌ No 'wiki*.json' files found in {LAYOUT_DIR}. Please check the path.")
        exit(1)

    print(f"Found {len(wiki_files)} wiki files to process.")

    with open(OUTPUT_FILE_PATH, 'a', encoding='utf-8') as f_out:
        
        for wiki_file in tqdm(wiki_files, desc="Processing Wiki Files", position=0):
            
            wiki_id = wiki_file.stem.replace("wiki", "") 
            
            try:
                with open(wiki_file, 'r', encoding='utf-8') as f:
                    wiki_data_list = json.load(f)
            except json.JSONDecodeError:
                tqdm.write(f"Skipping {wiki_file}: Could not parse JSON.")
                continue

            for item in tqdm(wiki_data_list, desc=f"Wiki {wiki_id}", position=1, leave=False):
                
                layout_index = item.get('index')
                if layout_index is None: 
                    tqdm.write(f"Skipping item in {wiki_file}: Missing 'index'.")
                    continue
                
                layout_for_prompt = {
                    "layers_all": item.get("layers_all", []),
                    "full_image_caption": item.get("full_image_caption", "")
                }
                
                qas_to_process = []
                
                original_qas = item.get('original_qa_pairs', [])
                for qa in original_qas:
                    question = qa.get('question')
                    squad_id = qa.get('id')
                    answers_list = qa.get('answers', {}).get('text', [])
                    
                    if question and squad_id and answers_list:
                        qas_to_process.append({
                            "question": question,
                            "answer": answers_list[0],
                            "qa_id": squad_id,
                            "source": "original"
                        })

                generated_qas = item.get('generated_qa_pairs', []) 
                for i, qa in enumerate(generated_qas):
                    question = qa.get('question')
                    answer = qa.get('answer')
                    
                    if question and answer:
                        gen_id = f"gen_{layout_index}_{i+1}" # Tạo ID duy nhất
                        qas_to_process.append({
                            "question": question,
                            "answer": answer,
                            "qa_id": gen_id,
                            "source": "generated"
                        })

                if not qas_to_process:
                    tqdm.write(f"Skipping item {layout_index} in {wiki_file}: No valid QAs found.")
                    continue

                # Vòng lặp duy nhất xử lý tất cả QAs đã gộp
                for qa_data in qas_to_process:
                    question = qa_data["question"]
                    ground_truth_answer = qa_data["answer"]
                    qa_id = qa_data["qa_id"] # Dùng ID đã chuẩn hóa
                    
                    tqdm.write("\n" + "="*50)
                    tqdm.write(f"Processing: Wiki: {wiki_id}, Index: {layout_index}, QA ID: {qa_id} (Source: {qa_data['source']})")
                    tqdm.write(f"❓ Question: {question}")
                    tqdm.write(f"🎯 Ground Truth (Provided): {ground_truth_answer}")
                    tqdm.write("⏳ Generating reasoning chain ...")
                    
                    think, content = generate_reasoning_chain(
                        layout_data=layout_for_prompt,
                        question=question,
                        ground_truth_answer=ground_truth_answer
                    )
                    
                    tqdm.write(f"✅ Think Output (token 151668): {think}")
                    
                    try:
                        content_json = json.loads(content)
                        tqdm.write("✅ Reasoning JSON Output:\n")
                        tqdm.write(json.dumps(content_json, indent=2))
                        stitched_reasoning = stitch_reasoning_json(content_json)
                    except json.JSONDecodeError:
                        tqdm.write(f"❌ Error: Model output was not valid JSON. Raw output: {content}")
                        content_json = {"error": "Invalid JSON from model", "raw_output": content}
                        stitched_reasoning = "Error: Failed to decode model output."
                    
                    tqdm.write("\n🔍 Merged Reasoning:\n")
                    tqdm.write(stitched_reasoning)

                    result_item = {
                        "wiki_id": wiki_id,
                        "layout_index": layout_index,
                        "squad_id": qa_id, 
                        "question": question,
                        "ground_truth_answer": ground_truth_answer,
                        "generated_reasoning": content_json, 
                        "merged_reasoning": stitched_reasoning 
                    }
                    
                    f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')

    print("\n🎉 All wiki files processed successfully.")