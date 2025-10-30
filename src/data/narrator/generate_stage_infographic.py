import json
import os
import argparse
import time
import random
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
# Seed management utilities
# ======================

def set_random_seed(seed: Optional[int] = None) -> int:
    """Set random seed for reproducibility and return the seed used
    
    Sets seeds for all commonly used libraries in LLM training:
    - Python random
    - NumPy
    - PyTorch
    - CUDA (if available)
    - Transformers
    - Environment variables for additional determinism
    """
    if seed is None:
        seed = random.randint(1, 1000000)
    
    # Python random
    random.seed(seed)
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        
        # CUDA seeds
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
            
        # Additional PyTorch determinism settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    except ImportError:
        pass
    
    # Transformers library seed
    try:
        from transformers import set_seed as transformers_set_seed
        transformers_set_seed(seed)
    except ImportError:
        pass
    
    # Environment variables for additional determinism
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For deterministic CUDA operations
    
    # TensorFlow (if used)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    return seed

# ======================
# Keyword checking utilities
# ======================

def extract_answer_keywords(qa_pairs: List[Dict]) -> List[str]:
    """Extract all answer keywords from QA pairs, excluding impossible questions"""
    keywords = []
    for qa in qa_pairs:
        answers = qa.get('answers', {})
        
        # Skip questions without answers (Squad v2 impossible questions)
        if not answers or not answers.get('text') or len(answers.get('text', [])) == 0:
            continue
            
        # Also check if the answer is empty or only contains empty strings
        if 'text' in answers and isinstance(answers['text'], list):
            valid_answers = [ans for ans in answers['text'] if ans and ans.strip()]
            if not valid_answers:  # Skip if no valid answers
                continue
                
            for answer_text in valid_answers:
                # Keep original case for exact matching
                keywords.append(answer_text.strip())
    return keywords

def has_answerable_questions(qa_pairs: List[Dict]) -> bool:
    """Check if there are any answerable questions in the QA pairs"""
    for qa in qa_pairs:
        answers = qa.get('answers', {})
        
        # Check if this question has valid answers
        if answers and answers.get('text') and len(answers.get('text', [])) > 0:
            if 'text' in answers and isinstance(answers['text'], list):
                valid_answers = [ans for ans in answers['text'] if ans and ans.strip()]
                if valid_answers:  # Found at least one valid answer
                    return True
    return False

def check_keywords_in_caption(caption: str, keywords: List[str]) -> tuple[bool, List[str]]:
    """Check if any of the keywords appear in the caption (case insensitive)"""
    if not caption or not keywords:
        return False, []
    
    caption_lower = caption.lower()
    found_keywords = []
    
    for keyword in keywords:
        if keyword.lower() in caption_lower:
            found_keywords.append(keyword)
    
    return len(found_keywords) > 0, found_keywords

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
    infographic_id: int,
    max_retries: int = 2
) -> Optional[Dict[str, Any]]:
    """
    Process a single sample through all 3 stages with keyword checking and retry logic
    
    Args:
        qwen_inference: Qwen3Inference instance
        item: Input data item (should contain 'context' and 'qa_pairs')
        stage_a_path: Path to Stage 1 template
        stage_b_path: Path to Stage 2 template  
        stage_c_path: Path to Stage 3 template
        infographic_id: Unique infographic ID
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with processed results or None if keywords not found after retries
    """
    # Ensure template files exist
    ensure_file(stage_a_path, "Stage 1")
    ensure_file(stage_b_path, "Stage 2")
    ensure_file(stage_c_path, "Stage 3")

    stage_a_tmpl_text = read_text(stage_a_path)
    stage_b_tmpl_text = read_text(stage_b_path)
    stage_c_tmpl_text = read_text(stage_c_path)

    # Extract keywords from QA pairs
    qa_pairs = item.get('qa_pairs', [])
    keywords = extract_answer_keywords(qa_pairs)
    
    # Check if there are any answerable questions
    has_answers = has_answerable_questions(qa_pairs)
    
    if not has_answers:
        print(f"Processing infographic {infographic_id}, No answerable questions (Squad v2 impossible questions) - skipping keyword check")
    else:
        print(f"Processing infographic {infographic_id}, Keywords to check: {keywords}")

    for retry_count in range(max_retries + 1):  # +1 because we include the first attempt
        try:
            # Set different seed for each retry attempt
            current_seed = set_random_seed()
            if retry_count > 0:
                print(f"  Retry attempt {retry_count}/{max_retries} with seed {current_seed}")

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

            # Check if keywords are present in the final caption
            # Skip keyword checking if there are no answerable questions (Squad v2 impossible questions)
            if not has_answers:
                # No answerable questions, accept the result without keyword checking
                keywords_found = True
                found_keywords = []
                print(f"  ✓ No answerable questions - accepting result without keyword check")
            else:
                keywords_found, found_keywords = check_keywords_in_caption(final_desc, keywords)
            
            if keywords_found or not keywords:  # Success if keywords found or no keywords to check
                if has_answers:
                    print(f"  ✓ Keywords found: {found_keywords}")
                
                # Combined result in format compatible with generate_infographic_data.py
                result = {
                    "id": item.get("id", None),
                    "title": item.get("title", None),
                    "context": context,
                    "qa_pairs": qa_pairs,
                    "keywords": keywords,
                    "keywords_found": found_keywords,
                    "has_answerable_questions": has_answers,
                    "skipped_keyword_check": not has_answers,
                    "retry_count": retry_count,
                    "final_seed": current_seed,
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
            else:
                print(f"  ✗ Keywords not found in caption. Required: {keywords}")
                if retry_count < max_retries:
                    print(f"    Retrying with different seed...")
                    continue
                else:
                    print(f"    Max retries ({max_retries}) reached. Returning None.")
                    return None
                    
        except Exception as e:
            print(f"  ✗ Error in attempt {retry_count + 1}: {str(e)}")
            if retry_count < max_retries:
                print(f"    Retrying due to error...")
                continue
            else:
                # Return error result after all retries failed
                return {
                    "id": item.get("id", None),
                    "title": item.get("title", None),
                    "context": item.get('context', ''),
                    "qa_pairs": qa_pairs,
                    "keywords": keywords,
                    "keywords_found": [],
                    "has_answerable_questions": has_answers,
                    "skipped_keyword_check": not has_answers,
                    "retry_count": retry_count,
                    "final_seed": None,
                    "generated_infographic": None,
                    "success": False,
                    "infographic_id": infographic_id,
                    "error": str(e)
                }
    
    # This should never be reached, but just in case
    return None

# ======================
# Data loading (from JSONL file)
# ======================

def load_squad_v2_data(input_path: str) -> List[Dict[str, Any]]:
    """Load Squad v2 data from JSONL file with context deduplication, keeping all QA pairs for each unique context"""
    all_data = []
    seen_contexts = {}  # Map context to list of QA pairs
    total_entries = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            total_entries += 1
            
            # Only keep entries with context
            if 'context' in item:
                context = item['context']
                
                # Create QA pair info
                qa_info = {
                    'question': item.get('question', ''),
                    'answers': item.get('answers', {}),
                    'id': item.get('id', None),
                    'title': item.get('title', None)
                }
                
                if context in seen_contexts:
                    # Add this QA pair to existing context
                    seen_contexts[context]['qa_pairs'].append(qa_info)
                else:
                    # First time seeing this context
                    seen_contexts[context] = {
                        'context': context,
                        'qa_pairs': [qa_info]
                    }
    
    # Convert to list format
    for context_data in seen_contexts.values():
        all_data.append(context_data)
    
    total_qa_pairs = sum(len(item['qa_pairs']) for item in all_data)
    print(f"Loaded {total_entries} total entries from Squad v2 file: {input_path}")
    print(f"Unique contexts: {len(all_data)} (deduplication removed {total_entries - len(all_data)} entries)")
    print(f"Total QA pairs: {total_qa_pairs}")
    return all_data

# ======================
# File saving (compatible with generate_infographic_data.py)
# ======================

def save_chunk_to_file(chunk: List[Optional[Dict]], output_dir: str, file_index: int) -> Optional[str]:
    """Save a chunk of results to file"""
    if not chunk:
        return None
    
    # Filter out None results (failed keyword checks)
    valid_results = [result for result in chunk if result is not None]
    
    if not valid_results:
        print(f"  ✗ No valid results to save for file {file_index:06d}")
        return None
    
    filename = f"infographic{file_index:06d}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(valid_results, f, ensure_ascii=False, indent=2)
    
    none_count = len(chunk) - len(valid_results)
    none_info = f" ({none_count} failed keyword checks)" if none_count > 0 else ""
    
    print(f"  ✓ Saved {len(valid_results)} infographics to {filename}{none_info}")
    if valid_results:
        print(f"    IDs: {valid_results[0]['infographic_id']}-{valid_results[-1]['infographic_id']}")
    
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
    parser.add_argument('--max_retries', type=int, default=2,
                        help='Maximum number of retry attempts when keywords not found (default: 2)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Initializing 3-Stage Infographic Data Generation")
    print("="*60)
    
    # Load input data
    print(f"\n[1/4] Loading input data from: {args.input_data}")
    input_data_full = load_squad_v2_data(args.input_data)
    print(f"Total unique contexts loaded: {len(input_data_full)}")
    
    # Print sample keywords for verification
    if input_data_full:
        sample_item = input_data_full[0]
        sample_keywords = extract_answer_keywords(sample_item.get('qa_pairs', []))
        print(f"Sample keywords from first context: {sample_keywords[:5]}")  # Show first 5

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
    print(f"Max retries: {args.max_retries}")
    print("-"*60)
    
    # Initialize variables for incremental saving
    results = []
    saved_files = []
    total_processed = 0
    successful_count = 0
    failed_count = 0
    keyword_failed_count = 0
    skipped_keyword_check_count = 0

    for i, item in enumerate(tqdm(input_data, desc="Processing samples")):
        # Calculate global infographic_id based on start data index
        infographic_id = start_data_idx + i + 1
        
        # Process single sample through 3-stage pipeline with keyword checking
        result = process_sample(
            qwen_inference,
            item,
            args.stage_a,
            args.stage_b,
            args.stage_c,
            infographic_id,
            args.max_retries
        )
        
        if result is None:
            # Keywords not found after all retries
            keyword_failed_count += 1
            failed_count += 1
        elif result["success"]:
            successful_count += 1
            # Count cases where keyword check was skipped
            if result.get("skipped_keyword_check", False):
                skipped_keyword_check_count += 1
        else:
            # Other errors (exception during processing)
            failed_count += 1
            
        results.append(result)  # Can be None
        total_processed += 1
        
        # Save to file when we have enough results
        if len(results) >= chunk_size:
            # Calculate file index based on first non-None infographic_id in chunk
            valid_results = [r for r in results if r is not None]
            if valid_results:
                first_infographic_id = valid_results[0]['infographic_id']
                file_index = (first_infographic_id - 1) // chunk_size + 1
            else:
                # If all results are None, use the expected file index
                file_index = (start_data_idx + i - len(results) + 2) // chunk_size + 1
            
            filename = save_chunk_to_file(results, args.output_dir, file_index)
            if filename:
                saved_files.append(filename)
            results = []  # Clear the results list
    
    # Save any remaining results
    if results:
        print("\n" + "="*60)
        print("Saving final chunk")
        print("="*60)
        # Calculate file index based on first non-None infographic_id in chunk
        valid_results = [r for r in results if r is not None]
        if valid_results:
            first_infographic_id = valid_results[0]['infographic_id']
            file_index = (first_infographic_id - 1) // chunk_size + 1
        else:
            # If all results are None, use the expected file index
            file_index = (start_data_idx + total_processed - len(results) + 1) // chunk_size + 1
        
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
    print(f"  - With keyword check: {successful_count - skipped_keyword_check_count} ({(successful_count - skipped_keyword_check_count)/total_processed*100:.1f}%)")
    print(f"  - Skipped keyword check (no answers): {skipped_keyword_check_count} ({skipped_keyword_check_count/total_processed*100:.1f}%)")
    print(f"Failed (errors): {failed_count - keyword_failed_count} ({(failed_count - keyword_failed_count)/total_processed*100:.1f}%)")
    print(f"Failed (keywords not found): {keyword_failed_count} ({keyword_failed_count/total_processed*100:.1f}%)")
    print(f"Total failed: {failed_count} ({failed_count/total_processed*100:.1f}%)")
    print(f"Output directory: {args.output_dir}")
    
    # Save a summary of failed cases if any
    if failed_count > 0:
        print(f"\nNote: Failed cases were distributed across multiple files.")
        print(f"Check individual files for 'success': false entries.")
        print(f"Keyword failures result in None entries (not saved to files).")
    
    if skipped_keyword_check_count > 0:
        print(f"\nNote: {skipped_keyword_check_count} samples had no answerable questions (Squad v2 impossible questions)")
        print(f"These samples were processed successfully without keyword checking.")
    
    print(f"\nFiles saved:")
    for filename in saved_files:
        print(f"  - {filename}")
    
    print("\n" + "="*60)
    print("3-Stage Generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()