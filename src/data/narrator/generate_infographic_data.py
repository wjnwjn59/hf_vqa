import json
import os
import re
from pathlib import Path
from jinja2 import Template
from tqdm import tqdm
import argparse
from typing import List, Dict, Any, Set, Optional

# Import the Qwen3 inference module
from src.inference.qwen3_inference import Qwen3Inference

# OpenAI client (lazy import to keep qwen-only runs lightweight)
try:
    from openai import OpenAI  # pip install openai
except Exception:
    OpenAI = None  # handled below

# ---------------------------
# Backend-agnostic inference
# ---------------------------

class OpenAIInference:
    """
    Minimal wrapper to mirror Qwen3Inference.generate() API:
      - generate(prompts: List[str], enable_thinking: bool=False) -> List[str]
    Uses the OpenAI Responses API, one call per prompt for simplicity/reliability.
    """
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
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
        self.max_tokens = max_tokens

    def generate(self, prompts: List[str], enable_thinking: bool = False) -> List[str]:
        outputs: List[str] = []
        for prompt in prompts:
            # You can optionally add a system message here if you have one.
            messages = [{"role": "user", "content": prompt}]
            resp = self.client.responses.create(
                model=self.model,
                input=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_output_tokens=self.max_tokens,
            )
            outputs.append((resp.output_text or "").strip())
        return outputs


def load_bizgen_template(template_path):
    """Load the bizgen.jinja template"""
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    return Template(template_content)

def extract_keywords_from_answers(qa_list: List[Dict]) -> Set[str]:
    """
    Extract keywords from answers for validation.
    """
    keywords = set()
    for qa in qa_list:
        answer = qa.get('answer', '')
        if not answer:
            continue
        answer_text = ""
        if isinstance(answer, dict):
            if 'text' in answer:
                if isinstance(answer['text'], list) and answer['text']:
                    answer_text = answer['text'][0]
                else:
                    answer_text = str(answer['text'])
        elif isinstance(answer, list):
            if answer:
                answer_text = str(answer[0])
        else:
            answer_text = str(answer)

        if not answer_text:
            continue

        answer_text = answer_text.lower()
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'should', 'could', 'may', 'might', 'must', 'can',
                      'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                      'into', 'through', 'during', 'before', 'after', 'above', 'below',
                      'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                      'under', 'again', 'further', 'then', 'once'}
        words = re.findall(r'\b[a-z0-9]+\b', answer_text)
        for word in words:
            if len(word) > 2 and word not in stop_words:
                keywords.add(word)
            elif word.isdigit():
                keywords.add(word)
    return keywords

def validate_keywords_in_output(output: str, keywords: Set[str], threshold: float = 0.3) -> bool:
    """
    Check if the output contains enough keywords from answers.
    """
    if not keywords:
        return True
    output_lower = output.lower()
    found_keywords = sum(1 for keyword in keywords if keyword in output_lower)
    coverage = found_keywords / len(keywords)
    return coverage >= threshold

def group_qa_by_context(data):
    """
    Group QA pairs by context.
    """
    context_groups = {}
    for item in data:
        context = item['context']
        qa_pair = {
            'question': item['question'],
            'answer': item['answer']
        }
        if context not in context_groups:
            context_groups[context] = {
                'context': context,
                'qa_list': [],
                'ids': [],
                'title': item.get('title', None)
            }
        context_groups[context]['qa_list'].append(qa_pair)
        if item.get('id'):
            context_groups[context]['ids'].append(item['id'])

    grouped_data = []
    for context, group_info in context_groups.items():
        group_info['qa_count'] = len(group_info['qa_list'])
        grouped_data.append(group_info)

    print(f"Grouped {len(data)} QA pairs into {len(grouped_data)} unique contexts")
    print(f"Average QA pairs per context: {len(data) / len(grouped_data):.2f}")
    return grouped_data

def load_input_data(dataset_type, input_path, deduplicate_context=False, group_by_context=False):
    """
    Load input data based on dataset type.
    """
    if dataset_type == "squad_v2":
        all_data = []
        seen_contexts = set()
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if 'context' in item and 'question' in item:
                    context = item['context']
                    if deduplicate_context and not group_by_context:
                        if context in seen_contexts:
                            continue
                        seen_contexts.add(context)
                    all_data.append({
                        'context': context,
                        'question': item['question'],
                        'answer': item.get('answers', ''),
                        'id': item.get('id', None),
                        'title': item.get('title', None)
                    })
        if group_by_context:
            all_data = group_qa_by_context(all_data)
            print(f"Final grouped data: {len(all_data)} unique contexts")
        elif deduplicate_context:
            print(f"Loaded {len(all_data)} unique contexts from Squad v2 file: {input_path}")
        else:
            print(f"Loaded {len(all_data)} entries from Squad v2 file: {input_path}")
        return all_data
    else:
        summarize_files = sorted([f for f in os.listdir(input_path) if f.startswith('summarize') and f.endswith('.json')])
        all_data = []
        for filename in summarize_files:
            filepath = os.path.join(input_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                all_data.extend(chunk_data)
        print(f"Loaded {len(all_data)} summarize entries from {len(summarize_files)} files")
        return all_data

def save_chunk_to_file(chunk, output_dir, file_index):
    """Save a chunk of results to file"""
    if not chunk:
        return None
    filename = f"infographic{file_index:06d}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved {len(chunk)} infographics to {filename} (IDs: {chunk[0]['infographic_id']}-{chunk[-1]['infographic_id']})")
    return filename

def main():
    parser = argparse.ArgumentParser(
        description='Generate infographic data with selectable backend (Qwen or OpenAI)'
    )
    # Backend selection
    parser.add_argument('--backend', type=str, default='qwen', choices=['qwen', 'openai'],
                        help="Inference backend: 'qwen' (local vLLM) or 'openai' (ChatGPT API)")

    # Qwen args
    parser.add_argument('--model-name', type=str, default='unsloth/Qwen3-8B',
                        help='Qwen model name or path (used when --backend qwen)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='GPU memory utilization (Qwen backend)')

    # OpenAI args
    parser.add_argument('--openai-model', type=str, default='gpt-4o',
                        help='OpenAI model name (used when --backend openai)')
    parser.add_argument('--openai-api-key', type=str, default=None,
                        help='OpenAI API key (optional; falls back to OPENAI_API_KEY env var)')

    # Data/template args
    parser.add_argument('--input-data', type=str,
                        default='/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl',
                        help='Path to input data file or directory')
    parser.add_argument('--dataset-type', type=str, default='squad_v2',
                        help='Type of dataset: squad_v2 or summarize')
    parser.add_argument('--template-path', type=str,
                        default='/home/thinhnp/hf_vqa/src/prompts/bizgen.jinja',
                        help='Path to bizgen.jinja template')
    parser.add_argument('--output-path', type=str,
                        default='/home/thinhnp/hf_vqa/src/data/create_data/qwen/infographic_generated.json',
                        help='Path to output JSON file')

    # Inference sampling args
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for inference (logical batching; OpenAI runs 1/call under the hood)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--max-tokens', type=int, default=8192,
                        help='Maximum tokens to generate')

    # Slicing/limiting
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to process (None for all)')
    parser.add_argument('--start', type=int, default=0,
                        help='Start index for data processing (inclusive)')
    parser.add_argument('--end', type=int, default=None,
                        help='End index for data processing (exclusive)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output dir for infographic files (default: src/data/create_data/output/infographic)')

    args = parser.parse_args()

    print("="*60)
    print("Initializing Infographic Data Generation")
    print("="*60)

    # Load template
    print(f"\n[1/4] Loading bizgen template from: {args.template_path}")
    template = load_bizgen_template(args.template_path)

    # Determine template behavior
    template_name = os.path.basename(args.template_path)
    deduplicate_context = template_name in ["bizgen_faithful_reproduction.jinja", "bizgen_design_drive_reproduction.jinja"]
    group_by_context = template_name == "content_des_all.jinja"
    requires_answer = template_name in ["bizgen_context_qa_full.jinja", "content_des_all.jinja"]

    if deduplicate_context:
        print(f"Template '{template_name}' detected - will deduplicate contexts")
    elif group_by_context:
        print(f"Template '{template_name}' detected - will group QA pairs by context")
    if requires_answer:
        print(f"Template '{template_name}' detected - will include answer field")

    # Load input data
    print(f"\n[2/4] Loading input data from: {args.input_data}")
    input_data_full = load_input_data(args.dataset_type, args.input_data,
                                      deduplicate_context=deduplicate_context,
                                      group_by_context=group_by_context)
    print(f"Total entries loaded: {len(input_data_full)}")

    # Apply slicing
    end_idx = args.end if args.end is not None else len(input_data_full)
    input_data_sliced = input_data_full[args.start:end_idx]
    print(f"Sliced data from index {args.start} to {end_idx}: {len(input_data_sliced)} samples")

    num_samples = args.num_samples if getattr(args, "num_samples", None) is not None else args.__dict__.get('num-samples')
    if num_samples:
        input_data = input_data_sliced[:num_samples]
        print(f"Further limited to {num_samples} samples")
    else:
        input_data = input_data_sliced
        print(f"Processing all {len(input_data)} samples from slice")

    # Initialize backend
    print(f"\n[3/4] Initializing inference backend: {args.backend}")
    if args.backend == 'qwen':
        print(f"Qwen model: {args.model_name}")
        inference = Qwen3Inference(
            model_name=args.model_name,
            max_model_len=32768,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="auto",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
    else:
        print(f"OpenAI model: {args.openai_model}")
        inference = OpenAIInference(
            model_name=args.openai_model,
            api_key=args.openai_api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )

    # Generate prompts
    print(f"\n[4/4] Generating infographic descriptions")
    print(f"Batch size: {args.batch_size}")
    print("-"*60)

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, '../../../'))
        output_dir = os.path.join(repo_root, 'src/data/create_data/output/infographic')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    results = []
    saved_files = []
    chunk_size = 50
    total_processed = 0
    successful_count = 0
    total_batches = (len(input_data) + args.batch_size - 1) // args.batch_size

    for batch_idx in tqdm(range(0, len(input_data), args.batch_size),
                          desc="Processing batches", total=total_batches):
        batch = input_data[batch_idx:batch_idx + args.batch_size]

        # Prepare prompts
        batch_prompts = []
        for item in batch:
            if args.dataset_type == "squad_v2":
                tname = os.path.basename(args.template_path)
                if tname in ["bizgen_faithful_reproduction.jinja", "bizgen_design_drive_reproduction.jinja"]:
                    rendered_prompt = template.render(paragraph_input=item["context"])
                elif tname == "bizgen_context_qa.jinja":
                    rendered_prompt = template.render(paragraph_input=item["context"], qa_pairs=item["question"])
                elif tname == "bizgen_context_qa_full.jinja":
                    rendered_prompt = template.render(context=item["context"],
                                                      question=item["question"],
                                                      answer=item["answer"])
                elif tname == "content_des_all.jinja":
                    rendered_prompt = template.render(context=item["context"],
                                                      qa_list=item["qa_list"])
                else:
                    rendered_prompt = template.render(paragraph_input=item["context"])
            else:
                rendered_prompt = template.render(brief_input=item.get("generated_summary", ""))
            batch_prompts.append(rendered_prompt)

        # Inference
        responses = inference.generate(batch_prompts, enable_thinking=False)

        # Process outputs
        for i, (item, response) in enumerate(zip(batch, responses)):
            infographic_id = args.start + total_processed + 1
            if group_by_context:
                keywords = extract_keywords_from_answers(item.get("qa_list", []))
            else:
                qa_list = [{"question": item.get("question", ""), "answer": item.get("answer", "")}]
                keywords = extract_keywords_from_answers(qa_list)

            keyword_coverage = validate_keywords_in_output(response, keywords, threshold=0.3)

            result = {
                "generated_infographic": response,
                "infographic_id": infographic_id,
                "keyword_coverage": keyword_coverage,
                "keywords_found": len([k for k in keywords if k in response.lower()]) if keywords else 0,
                "keywords_total": len(keywords)
            }

            if group_by_context:
                result.update({
                    "context": item["context"],
                    "qa_count": item.get("qa_count", len(item.get("qa_list", []))),
                    "ids": item.get("ids", []),
                    "title": item.get("title", None)
                })
            else:
                result.update({
                    "id": item.get("id", None),
                    "title": item.get("title", None)
                })

            successful_count += 1
            results.append(result)
            total_processed += 1

            if len(results) >= chunk_size:
                first_infographic_id = results[0]['infographic_id']
                file_index = (first_infographic_id - 1) // chunk_size + 1
                filename = save_chunk_to_file(results, output_dir, file_index)
                if filename:
                    saved_files.append(filename)
                results = []

    # Save any remaining results
    if results:
        print("\n" + "="*60)
        print("Saving final chunk")
        print("="*60)
        first_infographic_id = results[0]['infographic_id']
        file_index = (first_infographic_id - 1) // chunk_size + 1
        filename = save_chunk_to_file(results, output_dir, file_index)
        if filename:
            saved_files.append(filename)

    # Final statistics
    print("\n" + "="*60)
    print("Generation Complete - Final Statistics")
    print("="*60)
    if total_processed > 0:
        first_id = args.start + 1
        last_id = args.start + total_processed
        print(f"Infographic ID range: {first_id:06d} - {last_id:06d}")
    print(f"Total samples processed: {total_processed}")
    print(f"Total files saved: {len(saved_files)}")
    print(f"Successful: {successful_count}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles saved:")
    for filename in saved_files:
        print(f"  - {filename}")
    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
