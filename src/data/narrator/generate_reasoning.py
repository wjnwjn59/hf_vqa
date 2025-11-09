import os
import json
import argparse
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader

from src.inference.qwen3_inference import Qwen3Inference

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

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
    exit(1)


class OpenAIInference:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        if OpenAI is None:
            raise ImportError(
                "openai package not found. Please `pip install openai`."
            )
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "Missing OpenAI API key. Provide --openai_api_key or set OPENAI_API_KEY env var."
            )
        self.client = OpenAI(api_key=key)
        self.model = model_name
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, prompt: str, max_tokens: int) -> Tuple[str, str]:
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max_tokens
            )
            content = (resp.choices[0].message.content or "").strip()
            return ("", content)
        except Exception as e:
            print(f"❌ OpenAI API call failed: {e}")
            return ("", f"{{\"error\": \"API Error: {e}\"}}")


def generate_reasoning_qwen(
    layout_data: Dict[str, Any],
    question: str,
    ground_truth_answer: str,
    inference: Qwen3Inference,
    max_tokens: int
) -> Tuple[str, str]:
    layout_json_string = json.dumps(layout_data, indent=2)
    
    prompt = PROMPT_TEMPLATE_JINJA.render(
        layout_json_string=layout_json_string,
        question=question,
        answer=ground_truth_answer
    )

    response = inference.generate_single(
        prompt=prompt,
        enable_thinking=False,
        custom_sampling_params=None
    )

    response = response.strip()

    if "</think>" in response:
        think_end = response.find("</think>")
        thinking_content = response[:think_end].replace("<think>", "").strip()
        content = response[think_end + len("</think>"):].strip()
    else:
        thinking_content = ""
        content = response
    
    return thinking_content, content


def generate_reasoning_gpt(
    layout_data: Dict[str, Any],
    question: str,
    ground_truth_answer: str,
    client: OpenAIInference,
    max_tokens: int
) -> (str, str):
    
    layout_json_string = json.dumps(layout_data, indent=2)
    
    prompt = PROMPT_TEMPLATE_JINJA.render(
        layout_json_string=layout_json_string,
        question=question,
        answer=ground_truth_answer
    )
    
    return client.generate(prompt, max_tokens)


def norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    norm_x1 = max(0.0, min(x1 / w, 1.0))
    norm_y1 = max(0.0, min(y1 / h, 1.0))
    norm_x2 = max(0.0, min(x2 / w, 1.0))
    norm_y2 = max(0.0, min(y2 / h, 1.0))
    normalized_box = (round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3))
    return normalized_box


def generate_reasoning_json(data: Dict[str, Any], no_bbox: bool, no_spatial: bool) -> str:
    generated_understand = "Error: Could not parse dictionary."
    generated_think = "Error: Could not parse dictionary."
    generated_answer = "Error: No answer found."
    
    IMG_W, IMG_H = 1000, 1000
    
    try:
        element_data = {} # { "[ID]": {"desc": ..., "coords": ..., "content": ..., "spatial": ...} }

        # --- Pass 1: Lấy data từ 'understand' (description và coords) ---
        if "understand" in data and "relevant_elements" in data["understand"]:
            for el in data["understand"].get("relevant_elements", []):
                if 'id' in el:
                    key = f"[{el['id']}]"
                    element_data[key] = {
                        "desc": el.get('element_description'),
                        "coords": el.get('coordinates')
                    }
        
        # --- Pass 2: Lấy data từ 'think' (content và spatial), GHI ĐÈ/BỔ SUNG ---
        if "think" in data and "evidence_array" in data["think"]:
            for ev in data["think"].get("evidence_array", []):
                if 'id' in ev:
                    key = f"[{ev['id']}]"
                    if key not in element_data:
                        element_data[key] = {} # Tạo entry mới nếu chỉ có trong 'think'
                    
                    element_data[key].update({
                        "content": ev.get('content'),
                        "spatial": ev.get('spatial_context')
                    })

        # --- Pass 3: Xây dựng replacement_map từ data đã gộp ---
        replacement_map = {}
        for key, item in element_data.items():
            parts = []
            
            # 1. Xác định text: Ưu tiên 'content' (sạch hơn) từ 'think', nếu không có thì dùng 'desc' từ 'understand'
            text_base = item.get('content') or item.get('desc')
            
            if text_base and text_base.lower() != 'n/a':
                # Tìm text trong dấu ""
                quote_match = re.search(r'"(.*?)"', text_base)
                if quote_match:
                    parts.append(f"the text {quote_match.group(0)}")
                else:
                    # Nếu không có dấu "", dùng toàn bộ text_base làm mô tả
                    parts.append(f"the element '{text_base}'")
            
            # 2. Thêm coordinates (nếu có và không bị tắt)
            coords = item.get('coords')
            if coords and not no_bbox:
                norm_coords = norm_box_xyxy(coords, w=IMG_W, h=IMG_H)
                parts.append(f"at coordinates {list(norm_coords)}")
                
            # 3. Thêm spatial (nếu có và không bị tắt)
            spatial = item.get('spatial')
            if spatial and spatial.lower() != 'n/a' and not no_spatial:
                parts.append(f"which is {spatial}")
            
            replacement_map[key] = " ".join(parts) if parts else ""

        # --- Pass 4: Áp dụng replacement_map vào prose ---
        if "understand" in data and "analysis" in data["understand"]:
            generated_understand = data["understand"]["analysis"]
            for _ in range(5): # Lặp 5 lần để xử lý các ID lồng nhau
                replaced_in_pass = False
                for key, value in replacement_map.items():
                    if key in generated_understand:
                        generated_understand = generated_understand.replace(key, value)
                        replaced_in_pass = True
                if not replaced_in_pass:
                    break
        else:
            generated_understand = "Error: Missing 'understand' or 'analysis' structure."

        if "think" in data and "logical_reasoning" in data["think"]:
            generated_think = data["think"]["logical_reasoning"]
            for _ in range(5): # Lặp 5 lần để xử lý các ID lồng nhau
                replaced_in_pass = False
                for key, value in replacement_map.items():
                    if key in generated_think:
                        generated_think = generated_think.replace(key, value)
                        replaced_in_pass = True
                if not replaced_in_pass:
                    break
        else:
            generated_think = "Error: Missing 'think' or 'logical_reasoning' structure."

        generated_answer = data.get('answer', 'Error: Missing "answer" key.')

        # --- Pass 5: Dọn dẹp ---
        unreplaced_key_pattern = re.compile(r'\[\w+(?:_\w+)*\]') # Xóa các [ID] còn sót
        generated_understand = unreplaced_key_pattern.sub('', generated_understand)
        generated_think = unreplaced_key_pattern.sub('', generated_think)
        
        # **SỬA LỖI:** Chỉ xóa ( ) :
        cleanup_pattern = re.compile(r'[\(\):]') 
        generated_understand = cleanup_pattern.sub('', generated_understand)
        generated_think = cleanup_pattern.sub('', generated_think)

        generated_understand = re.sub(r'\s+', ' ', generated_understand).strip()
        generated_think = re.sub(r'\s+', ' ', generated_think).strip()
            
    except Exception as e:
        generated_understand = f"Error during generating: {e}"
        generated_think = f"Error during generating: {e}"
        generated_answer = f"Error during generating: {e}"

    return (
        f"{generated_understand} "
        f"{generated_think} "
        f"Therefore, the answer is {generated_answer}."
    )


def generate_short_reasoning(data: Dict[str, Any]) -> str:
    try:
        final_answer = data.get('answer', 'Error: Missing "answer" key.')
        evidence_texts = []

        if "think" in data and "evidence_array" in data["think"]:
            for ev in data["think"].get("evidence_array", []):
                content = ev.get('content')
                if content and content.lower() != 'n/a':
                    # **SỬA LỖI:** Dùng regex để tìm text, bất kể "Text "
                    quote_match = re.search(r'"(.*?)"', content)
                    if quote_match:
                        evidence_texts.append(quote_match.group(0)) # group(0) bao gồm cả dấu ""

        if not evidence_texts:
            return f"Based on the visual evidence in the image, the answer is {final_answer}."

        if len(evidence_texts) == 1:
            evidence_str = f"the text {evidence_texts[0]}"
        else:
            evidence_str = f"the texts {', '.join(evidence_texts)}"
        
        return f"Based on the image, {evidence_str} contain(s) the key information, so the answer is {final_answer}."

    except Exception as e:
        return f"Error during short reasoning generation: {e}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Reasoning with selectable backend (Qwen or GPT)'
    )
    parser.add_argument('--backend', type=str, default='qwen', choices=['qwen', 'gpt'],
                        help="Inference backend: 'qwen' (local) or 'gpt' (OpenAI API)")
    parser.add_argument('--model_name', type=str, default='/mnt/dataset1/pretrained_fm/Qwen_Qwen3-8B',
                        help='Qwen model name or path (used when --backend qwen)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization (Qwen backend with vLLM)')
    parser.add_argument('--max_model_len', type=int, default=24576,
                        help='Maximum model length for vLLM (Qwen backend)')
    parser.add_argument('--openai_model', type=str, default='gpt-4o',
                        help='OpenAI model name (used when --backend gpt)')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key (falls back to OPENAI_API_KEY env var)')
    parser.add_argument('--layout_dir', type=str,
                        default='/home/binhdt/hf_vqa/src/data/narrator/wiki_new',
                        help='Path to source directory for wiki*.json files')
    parser.add_argument('--output_file_path', type=str,
                        default='src/data/narrator/generated_reasonings.jsonl',
                        help='Path to the output .jsonl file')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=8096,
                        help='Maximum new tokens to generate')
    
    args = parser.parse_args()

    generate_fn = None
    if args.backend == 'qwen':
        print(f"🚀 Initializing Qwen model with vLLM: '{args.model_name}'...")
        inference = Qwen3Inference(
            model_name=args.model_name,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="auto",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        print("✅ Qwen model loaded successfully with vLLM.")
        
        generate_fn = lambda layout, q, a: generate_reasoning_qwen(
            layout, q, a, inference, args.max_tokens
        )
        
    else: 
        print(f"🚀 Initializing OpenAI client for model '{args.openai_model}'...")
        try:
            client = OpenAIInference(
                model_name=args.openai_model,
                api_key=args.openai_api_key,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print("✅ OpenAI client initialized successfully.")
            
            generate_fn = lambda layout, q, a: generate_reasoning_gpt(
                layout, q, a, client, args.max_tokens
            )
            
        except (ImportError, ValueError) as e:
            print(f"❌ {e}")
            exit(1)

    LAYOUT_DIR = Path(args.layout_dir)
    OUTPUT_FILE_PATH = Path(args.output_file_path)

    print("\n" + "="*80)
    print(f"🚀 Starting Reasoning generation using [Backend: {args.backend}]")
    print(f"Source Directory: {LAYOUT_DIR}")
    print(f"💾 Saving results to: {OUTPUT_FILE_PATH} (appending)")
    print("ℹ️  Generating all 5 reasoning versions (full, 3 ablations, 1 short) in a single run.")

    processed_qas = set()
    if OUTPUT_FILE_PATH.exists():
        print(f"📂 Loading existing results from {OUTPUT_FILE_PATH}...")
        try:
            with open(OUTPUT_FILE_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        existing_item = json.loads(line.strip())
                        wiki_id = existing_item.get('wiki_id')
                        layout_index = existing_item.get('layout_index')
                        qa_id = existing_item.get('squad_id')
                        if wiki_id is not None and layout_index is not None and qa_id is not None:
                            processed_qas.add((str(wiki_id), int(layout_index), str(qa_id)))
                    except json.JSONDecodeError:
                        continue
            print(f"✅ Found {len(processed_qas)} already processed QAs. Will skip them.")
        except Exception as e:
            print(f"⚠️  Warning: Could not read existing output file: {e}")
    else:
        print("📝 Output file does not exist yet. Will create new file.")
    
    wiki_files = sorted(list(LAYOUT_DIR.glob("wiki*.json")))
    
    if not wiki_files:
        print(f"❌ No 'wiki*.json' files found in {LAYOUT_DIR}. Please check the path.")
        exit(1)

    print(f"Found {len(wiki_files)} wiki files to process.")
    print("="*80 + "\n")

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
                        gen_id = f"gen_{layout_index}_{i+1}"
                        qas_to_process.append({
                            "question": question,
                            "answer": answer,
                            "qa_id": gen_id,
                            "source": "generated"
                        })

                if not qas_to_process:
                    continue

                for qa_data in qas_to_process:
                    question = qa_data["question"]
                    ground_truth_answer = qa_data["answer"]
                    qa_id = qa_data["qa_id"]
                    
                    qa_key = (str(wiki_id), int(layout_index), str(qa_id))
                    if qa_key in processed_qas:
                        continue
                    
                    tqdm.write("\n" + "="*50)
                    tqdm.write(f"Processing: Wiki: {wiki_id}, Index: {layout_index}, QA ID: {qa_id} (Source: {qa_data['source']})")
                    tqdm.write(f"❓ Question: {question}")
                    tqdm.write(f"🎯 Ground Truth (Provided): {ground_truth_answer}")
                    tqdm.write(f"⏳ Generating reasoning chain (Backend: {args.backend})...")
                    
                    think, content = generate_fn(
                        layout_for_prompt,
                        question,
                        ground_truth_answer
                    )
                    
                    tqdm.write(f"✅ Think Output: {think}")
                    
                    try:
                        content_json = json.loads(content)
                        tqdm.write("✅ Reasoning JSON Output:\n")
                        tqdm.write(json.dumps(content_json, indent=2))
                        
                        reasoning_full = generate_reasoning_json(
                            content_json, no_bbox=False, no_spatial=False
                        )
                        reasoning_no_bbox = generate_reasoning_json(
                            content_json, no_bbox=True, no_spatial=False
                        )
                        reasoning_no_spatial = generate_reasoning_json(
                            content_json, no_bbox=False, no_spatial=True
                        )
                        reasoning_short = generate_short_reasoning(content_json)
                        
                        tqdm.write("\n🔍 Reasoning Full:")
                        tqdm.write(reasoning_full)
                        tqdm.write("\n🔍 Reasoning No Bbox:")
                        tqdm.write(reasoning_no_bbox)
                        tqdm.write("\n🔍 Reasoning No Spatial:")
                        tqdm.write(reasoning_no_spatial)
                        tqdm.write("\n🔍 Reasoning Short:")
                        tqdm.write(reasoning_short)


                    except json.JSONDecodeError:
                        tqdm.write(f"❌ Error: Model output was not valid JSON. Raw output: {content}")
                        content_json = {"error": "Invalid JSON from model", "raw_output": content}
                        error_msg = "Error: Failed to decode model output."
                        reasoning_full = error_msg
                        reasoning_no_bbox = error_msg
                        reasoning_no_spatial = error_msg
                        reasoning_short = error_msg
                    
                    result_item = {
                        "wiki_id": wiki_id,
                        "layout_index": layout_index,
                        "squad_id": qa_id, 
                        "question": question,
                        "ground_truth_answer": ground_truth_answer,
                        "generated_reasoning": content_json, 
                        "reasoning_full": reasoning_full,
                        "reasoning_no_bbox": reasoning_no_bbox,
                        "reasoning_no_spatial": reasoning_no_spatial,
                        "reasoning_short": reasoning_short
                    }
                    
                    f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                    f_out.flush()
                    
                    processed_qas.add(qa_key)

    print("\n🎉 All wiki files processed successfully.")