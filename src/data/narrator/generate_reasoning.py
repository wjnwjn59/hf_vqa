import os
import torch
import json
import argparse # Đã thêm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader

# Import OpenAI (lazy import)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Cấu hình PyTorch CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

# --- Class Wrapper cho OpenAI ---
class OpenAIInference:
    """
    Wrapper tối giản để gọi API OpenAI,
    trả về (think, content) để tương thích.
    """
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
            # Trả về (think, content) để tương thích, GPT không có 'think'
            return ("", content)
        except Exception as e:
            print(f"❌ OpenAI API call failed: {e}")
            return ("", f"{{\"error\": \"API Error: {e}\"}}") # Trả về JSON lỗi

# --- Hàm sinh Reasoning cho Qwen (Giữ nguyên) ---
def generate_reasoning_qwen(
    layout_data: Dict[str, Any],
    question: str,
    ground_truth_answer: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_tokens: int
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
        max_new_tokens=max_tokens
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0 

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return thinking_content.strip(), content.strip()

# --- Hàm sinh Reasoning cho GPT ---
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
    
    # Client 'generate' đã trả về (think, content)
    return client.generate(prompt, max_tokens)


# --- Hàm Stitch (Giữ nguyên) ---
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

# --- Hàm Main Thực Thi ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Reasoning with selectable backend (Qwen or GPT)'
    )
    # Backend selection
    parser.add_argument('--backend', type=str, default='qwen', choices=['qwen', 'gpt'],
                        help="Inference backend: 'qwen' (local) or 'gpt' (OpenAI API)")

    # Qwen args
    parser.add_argument('--model_name', type=str, default='/mnt/dataset1/pretrained_fm/Qwen_Qwen3-8B',
                        help='Qwen model name or path (used when --backend qwen)')

    # GPT args
    parser.add_argument('--openai_model', type=str, default='gpt-4o',
                        help='OpenAI model name (used when --backend gpt)')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key (falls back to OPENAI_API_KEY env var)')

    # Data/template args
    parser.add_argument('--layout_dir', type=str,
                        default='/home/binhdt/hf_vqa/src/data/narrator/wiki',
                        help='Path to source directory for wiki*.json files')
    parser.add_argument('--output_file_path', type=str,
                        default='src/data/narrator/generated_reasonings.jsonl',
                        help='Path to the output .jsonl file')

    # Inference sampling args
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=32768,
                        help='Maximum new tokens to generate')
    
    args = parser.parse_args()

    # --- Khởi tạo Backend ---
    generate_fn = None
    if args.backend == 'qwen':
        print(f"🚀 Loading Qwen model '{args.model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✅ Qwen model and tokenizer loaded successfully.")
        
        # Tạo hàm generate tương thích
        generate_fn = lambda layout, q, a: generate_reasoning_qwen(
            layout, q, a, model, tokenizer, args.max_tokens
        )
        
    else: # args.backend == 'gpt'
        print(f"🚀 Initializing OpenAI client for model '{args.openai_model}'...")
        try:
            client = OpenAIInference(
                model_name=args.openai_model,
                api_key=args.openai_api_key,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print("✅ OpenAI client initialized successfully.")
            
            # Tạo hàm generate tương thích
            generate_fn = lambda layout, q, a: generate_reasoning_gpt(
                layout, q, a, client, args.max_tokens
            )
            
        except (ImportError, ValueError) as e:
            print(f"❌ {e}")
            exit(1)

    # --- Bắt đầu xử lý ---
    LAYOUT_DIR = Path(args.layout_dir)
    OUTPUT_FILE_PATH = Path(args.output_file_path)

    print("\n" + "="*80)
    print(f"🚀 Starting Reasoning generation using [Backend: {args.backend}]")
    print(f"Source Directory: {LAYOUT_DIR}")
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
                
                # 1. Thêm Original QAs
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

                # 2. Thêm Generated QAs
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
                    qa_id = qa_data["qa_id"] 
                    
                    tqdm.write("\n" + "="*50)
                    tqdm.write(f"Processing: Wiki: {wiki_id}, Index: {layout_index}, QA ID: {qa_id} (Source: {qa_data['source']})")
                    tqdm.write(f"❓ Question: {question}")
                    tqdm.write(f"🎯 Ground Truth (Provided): {ground_truth_answer}")
                    tqdm.write(f"⏳ Generating reasoning chain (Backend: {args.backend})...")
                    
                    # Gọi hàm generate_fn (Qwen hoặc GPT)
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