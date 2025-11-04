import os
import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    script_file_path = Path(__file__).resolve()
    src_dir = script_file_path.parent.parent.parent
except NameError:
    print("⚠️  '__file__' not defined. Assuming relative path for 'src'.")
    src_dir = Path.cwd().parent.parent 
    print(f"Set 'src_dir' to: {src_dir}")

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
            return ("", f"[] # API Error: {e}") 

def generate_qas_qwen(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_tokens: int
) -> tuple[str, str]:
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

def generate_qas_gpt(
    layout_data: Dict[str, Any],
    qa_samples: List[Dict[str, Any]], 
    k: int,
    client: OpenAIInference,
    max_tokens: int
) -> tuple[str, str]:
    """
    Hàm render prompt và gọi client OpenAI.
    """
    layout_json_string = json.dumps(layout_data, indent=2)
    
    prompt = PROMPT_TEMPLATE_JINJA.render(
        layout_json_string=layout_json_string,
        qa_samples=qa_samples,
        k=k
    )
    
    # Client 'generate' đã trả về (think, content)
    return client.generate(prompt, max_tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Question-Answers with selectable backend (Qwen or GPT)'
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
                        default='/home/binhdt/hf_vqa/src/data/wiki/',
                        help='Path to source/target directory for wiki*.json files')
    parser.add_argument('--k_value', type=int, default=3,
                        help='Number of new QAs to generate per item')

    # Inference sampling args
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=5000,
                        help='Maximum new tokens to generate')

    args = parser.parse_args()

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
        
        def qwen_fn_wrapper(layout, samples, k):
            layout_json_string = json.dumps(layout, indent=2)
            prompt = PROMPT_TEMPLATE_JINJA.render(
                layout_json_string=layout_json_string,
                qa_samples=samples,
                k=k
            )
            return generate_qas_qwen(prompt, model, tokenizer, args.max_tokens)
        
        generate_fn = qwen_fn_wrapper
        
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
            
            def gpt_fn_wrapper(layout, samples, k):
                return generate_qas_gpt(layout, samples, k, client, args.max_tokens)
                
            generate_fn = gpt_fn_wrapper
            
        except (ImportError, ValueError) as e:
            print(f"❌ {e}")
            exit(1)

    # --- Bắt đầu xử lý ---
    LAYOUT_DIR = Path(args.layout_dir)
    K_VALUE = args.k_value

    print("\n" + "="*80)
    print(f"🚀 Starting QA generation using [Backend: {args.backend}]")
    print(f"Source Directory (Read/Write): {LAYOUT_DIR}")
    print(f"New QAs to generate per item: {K_VALUE}")

    wiki_files = sorted(list(LAYOUT_DIR.glob("wiki*.json")))

    if not wiki_files:
        print(f"❌ No 'wiki*.json' files found in {LAYOUT_DIR}. Please check the path.")
        exit(1)

    print(f"Found {len(wiki_files)} wiki files to process.")
    
    
    for wiki_file in tqdm(wiki_files, desc="Processing Wiki Files", position=0):
        
        wiki_id = wiki_file.stem.replace("wiki", "")
        
        try:
            with open(wiki_file, 'r', encoding='utf-8') as f:
                wiki_data_list = json.load(f)
        except json.JSONDecodeError:
            tqdm.write(f"Skipping {wiki_file}: Could not parse JSON.")
            continue

        data_was_modified = False

        for item in tqdm(wiki_data_list, desc=f"Wiki {wiki_id}", position=1, leave=False):
            
            layout_index = item.get('index')
            if layout_index is None:
                tqdm.write(f"Skipping item in {wiki_file}: Missing 'index'.")
                continue
            
            layout_for_prompt = {
                "layers_all": item.get("layers_all", []),
                "full_image_caption": item.get("full_image_caption", "")
            }
            
            qa_samples_list = []
            original_qas = item.get('original_qa_pairs', [])
            
            if not original_qas:
                tqdm.write(f"Skipping ({wiki_id}, {layout_index}): No existing QAs found to use as samples.")
                continue

            for i, qa in enumerate(original_qas):
                question = qa.get('question')
                answers = qa.get('answers', {}).get('text', [])
                
                if question and answers:
                    qa_samples_list.append({
                        "id": i + 1,
                        "question": question,
                        "answer": answers[0]
                    })
            
            if not qa_samples_list:
                tqdm.write(f"Skipping ({wiki_id}, {layout_index}): Existing QAs were invalid.")
                continue

            tqdm.write("\n" + "="*50)
            tqdm.write(f"Processing: Wiki: {wiki_id}, Index: {layout_index}")
            tqdm.write(f"Found {len(qa_samples_list)} existing QAs to use as samples.")
            tqdm.write(f"⏳ Generating {K_VALUE} new QAs (Backend: {args.backend})...")
            
            thinking, content = generate_fn(
                layout_for_prompt,
                qa_samples_list,
                K_VALUE
            )
            
            tqdm.write(f"✅ Think Output: {thinking}")

            try:
                generated_qa_list = json.loads(content)
                if not isinstance(generated_qa_list, list):
                    tqdm.write(f"❌ Error: Model output was valid JSON but not a LIST. Skipping item.")
                    continue
                    
                tqdm.write("✅ Raw Generated JSON Array (Cleaned):\n")
                tqdm.write(json.dumps(generated_qa_list, indent=2))

                item['generated_qa_pairs'] = generated_qa_list 
                data_was_modified = True

            except json.JSONDecodeError:
                tqdm.write(f"❌ Error: Model output was not valid JSON. Raw output: {content}")
                continue

        if data_was_modified:
            tqdm.write(f"💾 Data modified. Overwriting file: {wiki_file}")
            try:
                with open(wiki_file, 'w', encoding='utf-8') as f_out:
                    json.dump(wiki_data_list, f_out, indent=2, ensure_ascii=False) 
            except Exception as e:
                tqdm.write(f"❌ FAILED to overwrite {wiki_file}: {e}")
        else:
            tqdm.write(f"No data modified for {wiki_file}. Skipping write.")

    print("\n🎉 All wiki files processed and updated successfully.")