import json
import os
import argparse
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from jinja2 import Template
from tqdm import tqdm

# Import the Qwen3 inference module
from src.inference.qwen3_inference import Qwen3Inference

# ======================
# I/O helpers
# ======================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def ensure_file(path: str, name: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required {name} template: {path}")

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def render_template(tmpl_text: str, **kwargs) -> str:
    return Template(tmpl_text, trim_blocks=True, lstrip_blocks=True).render(**kwargs).strip()

# ======================
# Sentence split (simplified from original)
# ======================

def split_into_sentences(context: str) -> List[str]:
    """Split context into sentences using simple regex fallback"""
    import re
    
    ctx = " ".join((context or "").strip().split())
    if not ctx:
        return []

    # Simple regex fallback (protect common abbreviations and decimals)
    ABBR_RE = re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|i\.e|e\.g|No)\.', flags=re.IGNORECASE)
    DECIMAL_RE = re.compile(r'(?<=\d)\.(?=\d)')
    PLACEHOLDER = '§DOT§'

    def _protect(s: str) -> str:
        s = ABBR_RE.sub(lambda m: m.group(0).replace('.', PLACEHOLDER), s)
        s = DECIMAL_RE.sub(PLACEHOLDER, s)
        return s

    def _restore(s: str) -> str:
        return s.replace(PLACEHOLDER, '.')

    prot = _protect(ctx)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"(\[]|$)', prot)
    out = [_restore(p).strip() for p in parts if p and p.strip()]
    return out

def enumerate_sentences(sents: List[str]) -> List[Dict[str, Any]]:
    return [{"id": i + 1, "text": s} for i, s in enumerate(sents)]

# ======================
# JSON extraction
# ======================

def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find('{')
    while start != -1:
        depth = 0
        i = start
        in_str = False
        esc = False
        while i < len(text):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break
            i += 1
        start = text.find('{', start + 1)
    return None

# ======================
# Parsers for Stage 1/2
# ======================

def parse_stage_a(text: str, sents_enum: List[Dict]) -> List[Dict[str, Any]]:
    js = extract_first_json(text)
    if not js or "summaries" not in js:
        # Fallback: create summaries from original sentences
        print("Warning: Stage 1 JSON parsing failed, using original sentences as summaries")
        items = []
        for sent in sents_enum:
            items.append({
                "id": sent["id"],
                "summary": sent["text"]
            })
        return items
    
    items = js["summaries"]
    for it in items:
        if "id" not in it or "summary" not in it:
            # Skip invalid items
            continue
        it["id"] = int(it["id"])
        it["summary"] = str(it["summary"]).strip()
    items.sort(key=lambda x: x["id"])
    return items

def parse_stage_b(text: str, summaries: List[Dict]) -> List[Dict[str, Any]]:
    js = extract_first_json(text)
    if not js or "figures" not in js:
        # Fallback: create generic figure ideas from summaries
        print("Warning: Stage 2 JSON parsing failed, using generic figure ideas")
        items = []
        for summary in summaries:
            items.append({
                "id": summary["id"],
                "ideas": ["A simple abstract illustration relevant to the content.", "A decorative graphic element."]
            })
        return items
    
    items = js["figures"]
    for it in items:
        if "id" not in it or "ideas" not in it:
            # Skip invalid items
            continue
        it["id"] = int(it["id"])
        if isinstance(it["ideas"], list):
            it["ideas"] = [str(s).strip() for s in it["ideas"] if str(s).strip()]
        else:
            it["ideas"] = [str(it["ideas"]).strip()]
        if len(it["ideas"]) == 0:
            it["ideas"] = ["A simple abstract illustration relevant to the sentence."]
        if len(it["ideas"]) > 2:
            it["ideas"] = it["ideas"][:2]
    items.sort(key=lambda x: x["id"])
    return items

# ======================
# Pipeline stages using Qwen3Inference
# ======================

def stage_a_summarize(qwen_inference: Qwen3Inference, sents_enum: List[Dict], stage_a_tmpl_text: str):
    prompt = render_template(stage_a_tmpl_text, sents=sents_enum)
    response = qwen_inference.generate_single(
        prompt,
        enable_thinking=False
    )
    return parse_stage_a(response, sents_enum)

def stage_b_figures(qwen_inference: Qwen3Inference, items: List[Dict], stage_b_tmpl_text: str):
    prompt = render_template(stage_b_tmpl_text, items=items)
    response = qwen_inference.generate_single(
        prompt,
        enable_thinking=False
    )
    return parse_stage_b(response, items)

def stage_c_compose(qwen_inference: Qwen3Inference, items: List[Dict], stage_c_tmpl_text: str, sents_enum: List[Dict]) -> str:
    prompt = render_template(stage_c_tmpl_text, items=items)
    response = qwen_inference.generate_single(
        prompt,
        enable_thinking=False
    )
    
    # Try to clean and use the response
    final_str = " ".join(response.strip().split())
    
    # If the response is too short or seems like an error, create a fallback caption
    if len(final_str) < 50 or not final_str:
        print("Warning: Stage 3 output too short, creating fallback caption")
        # Create a simple caption from the sentences
        sentence_texts = [sent["text"] for sent in sents_enum[:3]]  # Use first 3 sentences
        final_str = f"The image is an infographic that presents information about the following content: {' '.join(sentence_texts)}"
    
    return final_str

# ======================
# End-to-end per sample processing
# ======================

def process_sample(
    qwen_inference: Qwen3Inference,
    item: Dict[str, Any],
    stage_a_path: str,
    stage_b_path: str,
    stage_c_path: str,
    infographic_id: int
) -> Dict[str, Any]:
    """
    Process a single sample through all 3 stages
    
    Args:
        qwen_inference: Qwen3Inference instance
        item: Input data item (should contain 'context')
        stage_a_path: Path to Stage 1 template
        stage_b_path: Path to Stage 2 template  
        stage_c_path: Path to Stage 3 template
        infographic_id: Unique infographic ID
        
    Returns:
        Dictionary with processed results
    """
    # Ensure template files exist
    ensure_file(stage_a_path, "Stage 1")
    ensure_file(stage_b_path, "Stage 2")
    ensure_file(stage_c_path, "Stage 3")

    stage_a_tmpl_text = read_text(stage_a_path)
    stage_b_tmpl_text = read_text(stage_b_path)
    stage_c_tmpl_text = read_text(stage_c_path)

    try:
        # Stage 0: sentence segmentation
        context = item.get('context', '')
        sents = split_into_sentences(context)
        sents_enum = enumerate_sentences(sents)

        # Stage 1: summaries
        summaries = stage_a_summarize(qwen_inference, sents_enum, stage_a_tmpl_text)

        # Stage 2: figures
        figures = stage_b_figures(qwen_inference, summaries, stage_b_tmpl_text)

        # Merge 1 + 2 by id
        merged = {it["id"]: {"id": it["id"], "summary": it["summary"], "ideas": []} for it in summaries}
        for it in figures:
            if it["id"] in merged:
                merged[it["id"]]["ideas"] = it["ideas"]
        merged_items = [merged[k] for k in sorted(merged.keys())]

        # Stage 3: final caption
        final_desc = stage_c_compose(qwen_inference, merged_items, stage_c_tmpl_text, sents_enum)

        # Combined result in format compatible with generate_infographic_data.py
        result = {
            "id": item.get("id", None),
            "title": item.get("title", None),
            "generated_infographic": {
                "sentences": sents_enum,
                "summaries": summaries,
                "figures": merged_items,
                "full_image_caption": final_desc
            },
            "success": True,
            "infographic_id": infographic_id
        }
        
        return result
        
    except Exception as e:
        # Return error result in same format
        return {
            "id": item.get("id", None),
            "title": item.get("title", None),
            "generated_infographic": None,
            "success": False,
            "infographic_id": infographic_id,
            "error": str(e)
        }

# ======================
# Data loading (from JSONL file)
# ======================

def load_squad_v2_data(input_path: str) -> List[Dict[str, Any]]:
    """Load Squad v2 data from JSONL file with context deduplication"""
    all_data = []
    seen_contexts = set()
    total_entries = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            total_entries += 1
            
            # Only keep entries with context
            if 'context' in item:
                context = item['context']
                
                # Skip if we've already seen this context
                if context in seen_contexts:
                    continue
                    
                seen_contexts.add(context)
                all_data.append({
                    'context': context,
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'id': item.get('id', None),
                    'title': item.get('title', None)
                })
    
    print(f"Loaded {total_entries} total entries from Squad v2 file: {input_path}")
    print(f"Unique contexts: {len(all_data)} (deduplication removed {total_entries - len(all_data)} entries)")
    return all_data

# ======================
# File saving (compatible with generate_infographic_data.py)
# ======================

def save_chunk_to_file(chunk: List[Dict], output_dir: str, file_index: int) -> Optional[str]:
    """Save a chunk of results to file"""
    if not chunk:
        return None
    
    filename = f"infographic{file_index:06d}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Saved {len(chunk)} infographics to {filename} (IDs: {chunk[0]['infographic_id']}-{chunk[-1]['infographic_id']})")
    return filename

# ======================
# Main function
# ======================

def main():
    parser = argparse.ArgumentParser(description='Generate 3-stage infographic data using Qwen3 with vLLM')
    parser.add_argument('--model_name', type=str, default='unsloth/Qwen3-8B',
                        help='Model name or path')
    parser.add_argument('--input_data', type=str, 
                        default='/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl',
                        help='Path to Squad v2 JSONL file')
    parser.add_argument('--stage_a', type=str,
                        default='/home/thinhnp/hf_vqa/src/prompts/content_des_stage_1.jinja',
                        help='Path to Stage 1 Jinja template')
    parser.add_argument('--stage_b', type=str,
                        default='/home/thinhnp/hf_vqa/src/prompts/content_des_stage_2.jinja',
                        help='Path to Stage 2 Jinja template')
    parser.add_argument('--stage_c', type=str,
                        default='/home/thinhnp/hf_vqa/src/prompts/content_des_stage_3.jinja',
                        help='Path to Stage 3 Jinja template')
    parser.add_argument('--output_dir', type=str,
                        default='/home/thinhnp/hf_vqa/src/data/create_data/output/infographic_v2',
                        help='Output directory for infographic files')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=8192,
                        help='Maximum tokens to generate')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization')
    parser.add_argument('--start', type=int, default=1,
                        help='Start file index for data processing (inclusive, 1-based)')
    parser.add_argument('--end', type=int, default=None,
                        help='End file index for data processing (exclusive, 1-based)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (None for all)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Initializing 3-Stage Infographic Data Generation")
    print("="*60)
    
    # Load input data
    print(f"\n[1/4] Loading input data from: {args.input_data}")
    input_data_full = load_squad_v2_data(args.input_data)
    print(f"Total unique contexts loaded: {len(input_data_full)}")

    # Convert file indices to data indices (each file contains 50 unique contexts)
    chunk_size = 50
    start_data_idx = (args.start - 1) * chunk_size
    end_data_idx = (args.end - 1) * chunk_size if args.end is not None else len(input_data_full)
    
    # Validate file indices
    max_files_needed = (len(input_data_full) + chunk_size - 1) // chunk_size  # Round up
    if args.start < 1:
        raise ValueError("Start file index must be >= 1")
    if args.end is not None and args.end <= args.start:
        raise ValueError("End file index must be > start file index")
    if args.start > max_files_needed:
        raise ValueError(f"Start file index {args.start} exceeds available data (max files: {max_files_needed})")
    
    print(f"File indices: {args.start} to {args.end if args.end else 'end'}")
    print(f"Data indices: {start_data_idx} to {end_data_idx}")
    print(f"Unique contexts per file: {chunk_size}")

    # Apply start and end slicing first
    input_data_sliced = input_data_full[start_data_idx:end_data_idx]
    print(f"Sliced data from data index {start_data_idx} to {end_data_idx}: {len(input_data_sliced)} samples")

    # Then apply num_samples limit if specified
    if args.num_samples:
        input_data = input_data_sliced[:args.num_samples]
        print(f"Further limited to {args.num_samples} samples")
    else:
        input_data = input_data_sliced
        print(f"Processing all {len(input_data)} samples from slice")
    
    # Initialize Qwen3 inference model
    print(f"\n[2/4] Initializing Qwen3 inference model: {args.model_name}")
    qwen_inference = Qwen3Inference(
        model_name=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto",
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens * 2
    )
    
    # Create output directory
    print(f"\n[3/4] Setting up output directory: {args.output_dir}")
    ensure_dir(args.output_dir)
    
    # Process data
    print(f"\n[4/4] Processing samples through 3-stage pipeline")
    print(f"Templates:")
    print(f"  Stage 1: {args.stage_a}")
    print(f"  Stage 2: {args.stage_b}")
    print(f"  Stage 3: {args.stage_c}")
    print("-"*60)
    
    # Initialize variables for incremental saving
    results = []
    saved_files = []
    total_processed = 0
    successful_count = 0
    failed_count = 0

    for i, item in enumerate(tqdm(input_data, desc="Processing samples")):
        # Calculate global infographic_id based on start data index
        infographic_id = start_data_idx + i + 1
        
        # Process single sample through 3-stage pipeline
        result = process_sample(
            qwen_inference,
            item,
            args.stage_a,
            args.stage_b,
            args.stage_c,
            infographic_id
        )
        
        if result["success"]:
            successful_count += 1
        else:
            failed_count += 1
            
        results.append(result)
        total_processed += 1
        
        # Save to file when we have enough results
        if len(results) >= chunk_size:
            # Calculate file index based on first infographic_id in chunk
            first_infographic_id = results[0]['infographic_id']
            file_index = (first_infographic_id - 1) // chunk_size + 1
            filename = save_chunk_to_file(results, args.output_dir, file_index)
            if filename:
                saved_files.append(filename)
            results = []  # Clear the results list
    
    # Save any remaining results
    if results:
        print("\n" + "="*60)
        print("Saving final chunk")
        print("="*60)
        # Calculate file index based on first infographic_id in chunk
        first_infographic_id = results[0]['infographic_id']
        file_index = (first_infographic_id - 1) // chunk_size + 1
        filename = save_chunk_to_file(results, args.output_dir, file_index)
        if filename:
            saved_files.append(filename)
    
    # Final statistics
    print("\n" + "="*60)
    print("3-Stage Generation Complete - Final Statistics")
    print("="*60)
    
    if total_processed > 0:
        first_id = start_data_idx + 1
        last_id = start_data_idx + total_processed
        print(f"Infographic ID range: {first_id:06d} - {last_id:06d}")
        print(f"File index range: {args.start} - {args.end if args.end else args.start + (total_processed + chunk_size - 1) // chunk_size}")
    
    print(f"Total samples processed: {total_processed}")
    print(f"Total files saved: {len(saved_files)}")
    print(f"Successful: {successful_count} ({successful_count/total_processed*100:.1f}%)")
    print(f"Failed: {failed_count} ({failed_count/total_processed*100:.1f}%)")
    print(f"Output directory: {args.output_dir}")
    
    # Save a summary of failed cases if any
    if failed_count > 0:
        print(f"\nNote: Failed cases were distributed across multiple files.")
        print(f"Check individual files for 'success': false entries.")
    
    print(f"\nFiles saved:")
    for filename in saved_files:
        print(f"  - {filename}")
    
    print("\n" + "="*60)
    print("3-Stage Generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()