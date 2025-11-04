import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader

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


def generate_new_qas(
    layout_data: Dict[str, Any],
    qa_samples: List[Dict[str, Any]], 
    k: int                                  
) -> tuple[str, str]:
    
    layout_json_string = json.dumps(layout_data, indent=2)
    
    prompt = PROMPT_TEMPLATE_JINJA.render(
        layout_json_string=layout_json_string,
        qa_samples=qa_samples,
        k=k
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

if __name__ == '__main__':
    LAYOUT_DIR = Path("/home/binhdt/hf_vqa/src/data/wiki/")
    K_VALUE = 3 

    print("\n" + "="*80)
    print(f"🚀 Starting QA generation...")
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
            tqdm.write(f"⏳ Generating {K_VALUE} new QAs...")

            thinking, content = generate_new_qas(
                layout_data=layout_for_prompt,
                qa_samples=qa_samples_list,
                k=K_VALUE
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