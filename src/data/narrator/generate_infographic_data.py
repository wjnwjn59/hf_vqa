import json
import os
import re
from pathlib import Path
from jinja2 import Template
from tqdm import tqdm
import argparse
from typing import List, Dict, Any, Set, Optional, Tuple

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

    def generate(
        self,
        prompts: List[str],
        enable_thinking: bool = False,
        qa_lists: Optional[List[List[Dict]]] = None,
        max_retries: int = 3,
    ) -> List[str]:
        outputs: List[str] = []

        # -----------------------
        # Helpers (internal use)
        # -----------------------
        def _extract_keywords(qa_list: Optional[List[Dict]]) -> set:
            return extract_keywords_from_answers(qa_list, debug=False) if qa_list else set()

        def _eval_response(text: str, kws: set, qas: Optional[List[Dict]]):
            """Return tuple: (figure_count, missing_keywords, missing_pairs, keyword_coverage, score)"""
            figs = count_figures_in_output(text)
            missing_k, missing_pairs = find_missing_keyword_answers(text, kws, qas or [])
            coverage = 0.0
            if kws:
                found = len(kws) - len(missing_k)
                coverage = found / len(kws)
            score = (figs * 2.0) + (coverage * 10.0)
            return figs, missing_k, missing_pairs, coverage, score

        def _answer_text(answer: Any) -> str:
            if isinstance(answer, dict):
                val = answer.get("text", "")
                if isinstance(val, list):
                    return str(val[0]) if val else ""
                return str(val)
            if isinstance(answer, list):
                return str(answer[0]) if answer else ""
            return str(answer)

        def _build_feedback(figs: int, missing_pairs: List[Dict]) -> str:
            parts = []
            parts.append("Revision required: Please update the infographic caption based on the issues below.")
            if figs < 2:
                parts.append(
                    f"Figure issue: Your previous response only contained {figs} figure(s). "
                    f"You need at least 2 figures."
                )
            if missing_pairs:
                parts.append(
                    f"\nMissing QA information issue: {len(missing_pairs)} QA pair(s) were not covered in your previous response. "
                    "Please add concise new text items—short phrases of about 6–8 words—that explicitly include each missing answer "
                    "and integrate them into the current infographic content:"
                )
                for i, qa in enumerate(missing_pairs[:3], 1):
                    q = str(qa.get("question", "")).strip()
                    a = _answer_text(qa.get("answer", ""))
                    parts.append(f"{i}. Q: {q}\n   A: {a}")
            parts.append(
                "\nPlease regenerate the infographic description addressing all the above issues."
            )
            return "\n".join(parts)


        # -----------------------
        # Main loop
        # -----------------------
        for idx, prompt in enumerate(prompts):
            qa_list = qa_lists[idx] if qa_lists and idx < len(qa_lists) else []
            keywords = _extract_keywords(qa_list)

            messages = [{"role": "user", "content": prompt}]
            best_response = ""
            best_score = -1.0

            # ---- Initial generation ----
            resp = self.client.responses.create(
                model=self.model,
                input=messages,
                reasoning={"effort": "minimal"}
            )
            response_text = (resp.output_text or "").strip()
            if not response_text:
                outputs.append("")
                continue

            figs, miss_k, miss_pairs, cov, score = _eval_response(response_text, keywords, qa_list)
            best_response, best_score = response_text, score

            # If satisfied or no retries, return initial
            if (figs >= 2 and not miss_k) or max_retries <= 1:
                outputs.append(response_text)
                continue

            # ---- Seed feedback and enter retries ----
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": _build_feedback(figs, miss_pairs)})

            for retry in range(1, max_retries):
                resp = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    reasoning={"effort": "minimal"}
                )
                response_text = (resp.output_text or "").strip()

                if not response_text:
                    # Last attempt → keep best so far; otherwise ask to provide complete caption
                    if retry == max_retries - 1:
                        outputs.append(best_response if best_response else "")
                    else:
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": "Please provide a complete caption."})
                    continue

                figs, miss_k, miss_pairs, cov, score = _eval_response(response_text, keywords, qa_list)

                if score > best_score:
                    best_response, best_score = response_text, score

                if figs >= 2 and not miss_k:
                    outputs.append(response_text)
                    break

                if retry == max_retries - 1:
                    outputs.append(best_response if best_response else response_text)
                    break

                # Add feedback and continue
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": _build_feedback(figs, miss_pairs)})

        return outputs



# ---------------------------
# Helpers
# ---------------------------

def count_figures_in_output(output: str) -> int:
    pattern = r'\(figure\)'
    output_lower = output.lower()
    matches = re.findall(pattern, output_lower)
    return len(matches)

def load_bizgen_template(template_path):
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    return Template(template_content)

def extract_keywords_from_answers(qa_list: List[Dict], debug: bool = False) -> Set[str]:
    keywords = set()
    for qa_idx, qa in enumerate(qa_list):
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
        if debug:
            print(f"  QA {qa_idx + 1} - Original answer: {answer_text}")
        answer_text = answer_text.lower()
        stop_words = {'a','an','the','is','are','was','were','be','been','being','have','has','had',
                      'do','does','did','will','would','should','could','may','might','must','can',
                      'of','at','by','for','with','about','against','between','into','through','during',
                      'before','after','above','below','to','from','up','down','in','out','on','off','over',
                      'under','again','further','then','once'}
        words = re.findall(r'\b[a-z0-9]+\b', answer_text)
        qa_keywords = []
        for word in words:
            if len(word) > 2 and word not in stop_words:
                keywords.add(word); qa_keywords.append(word)
            elif word.isdigit():
                keywords.add(word); qa_keywords.append(word)
        if debug and qa_keywords:
            print(f"    → Extracted keywords: {qa_keywords}")
    return keywords

def validate_keywords_in_output(output: str, keywords: Set[str], threshold: float = 1.0, debug: bool = False) -> bool:
    if not keywords:
        if debug: print("  ✓ No keywords to validate, returning True")
        return True
    output_lower = output.lower()
    found_keywords, missing_keywords = [], []
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, output_lower):
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    coverage = len(found_keywords) / len(keywords)
    passed = coverage >= threshold
    if debug:
        print(f"\n  Keyword Validation Results:\n  {'='*60}")
        print(f"  Total keywords: {len(keywords)}")
        print(f"  Found keywords: {len(found_keywords)} ({coverage:.1%})")
        print(f"  Threshold: {threshold:.1%}")
        print(f"  Status: {'✓ PASSED' if passed else '✗ FAILED'}\n  {'='*60}\n")
    return passed

def find_missing_keyword_answers(output: str, keywords: Set[str], qa_list: List[Dict]) -> Tuple[Set[str], List[Dict]]:
    if not keywords:
        return set(), []
    output_lower = output.lower()
    missing_keywords = set()
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if not re.search(pattern, output_lower):
            missing_keywords.add(keyword)
    if not missing_keywords:
        return set(), []
    qa_pairs_with_missing = []
    for qa in qa_list:
        answer = qa.get('answer', '')
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
        answer_lower = answer_text.lower()
        for missing_kw in missing_keywords:
            pattern = r'\b' + re.escape(missing_kw) + r'\b'
            if re.search(pattern, answer_lower):
                qa_pairs_with_missing.append(qa); break
    return missing_keywords, qa_pairs_with_missing

def group_qa_by_context(data):
    context_groups = {}
    for item in data:
        context = item['context']
        qa_pair = {'question': item['question'], 'answer': item['answer']}
        if context not in context_groups:
            context_groups[context] = {
                'context': context, 'qa_list': [], 'ids': [], 'title': item.get('title', None)
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

# ============ NEW: failed regeneration helpers ============

def load_failed_cases(failed_path: str) -> List[Dict[str, Any]]:
    with open(failed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    print(f"Loaded {len(data)} failed cases from {failed_path}")
    return data

def build_failed_input_from_squad(failed_cases: List[Dict[str, Any]], squad_jsonl: str) -> List[Dict[str, Any]]:
    """
    Build a grouped-by-context input list from failed cases + SQuAD v2 jsonl.
    - If a failed case has 'ids': collect all QAs in the SQuAD file with those ids.
    - Else if it has 'id': collect that one QA.
    - Prefer context from the failed case when present; otherwise use context from SQuAD record.
    Returns a list of dicts: {context, qa_list:[{question,answer}], ids:[...], title, qa_count}
    """
    # Index SQuAD by id
    squad_by_id: Dict[str, Dict[str, Any]] = {}
    with open(squad_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rid = rec.get("id")
            if rid is not None:
                squad_by_id[str(rid)] = rec

    grouped: List[Dict[str, Any]] = []
    for fc in failed_cases:
        case_ids: List[str] = []
        if isinstance(fc.get("ids"), list) and fc["ids"]:
            case_ids = [str(x) for x in fc["ids"]]
        elif fc.get("id") is not None:
            case_ids = [str(fc["id"])]
        else:
            # No ids info; skip safely
            continue

        qa_list: List[Dict[str, Any]] = []
        context_from_failed = fc.get("context")
        context_from_squad = None
        title = fc.get("title")

        for cid in case_ids:
            rec = squad_by_id.get(str(cid))
            if not rec:
                continue
            if context_from_squad is None:
                context_from_squad = rec.get("context")
            qa_list.append({
                "question": rec.get("question", ""),
                "answer": rec.get("answers", "")
            })
            if title is None and rec.get("title"):
                title = rec.get("title")

        # Choose context: prefer failed (if present), else SQuAD
        context = context_from_failed or context_from_squad
        if not context or not qa_list:
            # skip if we can't reconstruct
            continue

        grouped.append({
            "context": context,
            "qa_list": qa_list,
            "ids": case_ids,
            "qa_count": len(qa_list),
            "title": title
        })

    # Optional: merge duplicates with the same context (rare for failed lists, but safe)
    if not grouped:
        print("Warning: No grouped inputs could be built from failed cases + SQuAD.")
    else:
        print(f"Prepared {len(grouped)} context groups for regeneration from failed cases.")
    return grouped
# =========================================================


def save_chunk_to_file(chunk, output_dir, file_index):
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
                        help='OpenAI API key (optional; falls back to OPENAI_API_KEY env var')

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
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retries for insufficient figures or missing keywords (OpenAI backend only)')

    # Slicing/limiting
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to process (None for all)')
    parser.add_argument('--start', type=int, default=1,
                        help='Start file index (1-based). --start 1 produces infographic000001.json with IDs 1-50')
    parser.add_argument('--end', type=int, default=None,
                        help='End file index (exclusive, 1-based). --start 1 --end 3 produces 2 files (000001, 000002)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output dir for infographic files (default: src/data/create_data/output/infographic)')
    parser.add_argument('--debug-keywords', action='store_true', default=False,
                        help='Enable detailed keyword validation logging')

    # NEW: failed regeneration options
    parser.add_argument('--regenerate-from-failed', type=str, default=None,
                        help='Path to failed.json (mismatched cases) to regenerate only those samples')
    parser.add_argument('--squad-jsonl', type=str, default=None,
                        help='Path to SQuAD v2 .jsonl used to reconstruct QAs for failed cases')

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

    # NEW: Regenerate-only mode from failed cases
    failed_mode = args.regenerate_from_failed is not None
    if failed_mode:
        if not args.squad_jsonl:
            raise ValueError("--squad-jsonl is required when using --regenerate-from-failed")
        print(f"\n[2/4] Regeneration mode: loading failed cases from {args.regenerate_from_failed}")
        failed_cases = load_failed_cases(args.regenerate_from_failed)
        input_data_full = build_failed_input_from_squad(failed_cases, args.squad_jsonl)
        # Force grouped-by-context behavior for consistent prompts
        group_by_context = True
        deduplicate_context = False
        print(f"Total entries prepared from failed cases: {len(input_data_full)}")
    else:
        # Load input data (standard path)
        print(f"\n[2/4] Loading input data from: {args.input_data}")
        input_data_full = load_input_data(args.dataset_type, args.input_data,
                                          deduplicate_context=deduplicate_context,
                                          group_by_context=group_by_context)
        print(f"Total entries loaded: {len(input_data_full)}")

    # Calculate how many samples to process
    if failed_mode:
        # In failed mode, we use the failed subset exactly; ignore start/end slicing
        input_data_sliced = input_data_full
        print(f"Regenerating {len(input_data_sliced)} failed contexts (slicing disabled).")
    else:
        if args.end is not None:
            num_files = args.end - args.start
            start_sample_idx = (args.start - 1) * 50
            end_sample_idx = start_sample_idx + (num_files * 50)
            input_data_sliced = input_data_full[start_sample_idx:end_sample_idx]
            print(f"File range: {args.start} to {args.end-1} ({num_files} files = {len(input_data_sliced)} samples)")
            print(f"Sample indices: {start_sample_idx} to {end_sample_idx-1}")
        else:
            start_sample_idx = (args.start - 1) * 50 if args.start >= 1 else 0
            end_sample_idx = len(input_data_full)
            input_data_sliced = input_data_full[start_sample_idx:end_sample_idx]
            print(f"Processing from file {args.start} onwards ({len(input_data_sliced)} samples)")
            print(f"Sample indices: {start_sample_idx} to {end_sample_idx-1}")

    # Optional limit
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
        max_retries = 1
    else:
        print(f"OpenAI model: {args.openai_model}")
        print(f"Max retries: {args.max_retries}")
        inference = OpenAIInference(
            model_name=args.openai_model,
            api_key=args.openai_api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        max_retries = args.max_retries

    # Generate
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
        batch_qa_lists = []
        for item in batch:
            if failed_mode or args.dataset_type == "squad_v2":
                tname = os.path.basename(args.template_path)
                if tname in ["bizgen_faithful_reproduction.jinja", "bizgen_design_drive_reproduction.jinja"]:
                    rendered_prompt = template.render(paragraph_input=item["context"])
                elif tname == "bizgen_context_qa.jinja":
                    rendered_prompt = template.render(paragraph_input=item["context"], qa_pairs=item["question"])
                elif tname == "bizgen_context_qa_full.jinja":
                    rendered_prompt = template.render(context=item["context"],
                                                      question=item.get("question",""),
                                                      answer=item.get("answer",""))
                elif tname == "content_des_all.jinja":
                    rendered_prompt = template.render(context=item["context"],
                                                      qa_list=item["qa_list"])
                else:
                    rendered_prompt = template.render(paragraph_input=item["context"])
            else:
                rendered_prompt = template.render(brief_input=item.get("generated_summary", ""))
            batch_prompts.append(rendered_prompt)

            if failed_mode or group_by_context:
                batch_qa_lists.append(item.get("qa_list", []))
            else:
                batch_qa_lists.append([{"question": item.get("question", ""), "answer": item.get("answer", "")}])

        # Inference
        if args.backend == 'openai':
            responses = inference.generate(batch_prompts, enable_thinking=False, 
                                           qa_lists=batch_qa_lists, max_retries=max_retries)
        else:
            responses = inference.generate(batch_prompts, enable_thinking=False)

        # Process outputs
        for i, (item, response) in enumerate(zip(batch, responses)):
            infographic_id = (args.start - 1) * 50 + total_processed + 1
            figure_count = count_figures_in_output(response)

            if args.debug_keywords:
                print(f"\n{'='*60}\nSample {infographic_id} - Keyword Extraction\n{'='*60}")

            if failed_mode or group_by_context:
                keywords = extract_keywords_from_answers(item.get("qa_list", []), debug=args.debug_keywords)
            else:
                qa_list = [{"question": item.get("question", ""), "answer": item.get("answer", "")}]
                keywords = extract_keywords_from_answers(qa_list, debug=args.debug_keywords)

            keyword_coverage = validate_keywords_in_output(response, keywords, threshold=1.0, debug=args.debug_keywords)

            # Count keyword hits (exact-word boundaries)
            keywords_found_count = 0
            if keywords:
                response_lower = response.lower()
                for k in keywords:
                    pattern = r'\b' + re.escape(k) + r'\b'
                    if re.search(pattern, response_lower):
                        keywords_found_count += 1

            result = {
                "generated_infographic": response,
                "infographic_id": infographic_id,
                "figure_count": figure_count,
                "keyword_coverage": keyword_coverage,
                "keywords_found": keywords_found_count,
                "keywords_total": len(keywords)
            }

            if failed_mode or group_by_context:
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

    # Save remaining
    if results:
        print("\n" + "="*60 + "\nSaving final chunk\n" + "="*60)
        first_infographic_id = results[0]['infographic_id']
        file_index = (first_infographic_id - 1) // chunk_size + 1
        filename = save_chunk_to_file(results, output_dir, file_index)
        if filename:
            saved_files.append(filename)

    # Final stats
    print("\n" + "="*60)
    print("Generation Complete - Final Statistics")
    print("="*60)
    if total_processed > 0:
        first_id = (args.start - 1) * 50 + 1
        last_id = (args.start - 1) * 50 + total_processed
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
