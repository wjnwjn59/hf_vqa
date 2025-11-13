from __future__ import annotations

import ast
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

PAIR_KEY_BY_TYPE = {
    "original": "original_qa_pairs",
    "generated": "generated_qa_pairs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert QA JSON files into conversation JSONL rows.")
    parser.add_argument(
        "--input",
        default="/data/thangdd_workspace/InfographicDataPaper/NarrativeInfoVQA/train_annotations.jsonl",
        help="Path to the input JSONL definition file.",
    )
    # If not provided, we will auto-name the output based on split/scope/reasoning.
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output JSONL. If omitted, will be narrativeinfovqa_{split}_{scope}_{reasoning}.jsonl",
    )
    parser.add_argument(
        "--has_reasoning",
        required=True,
        help="true/false: Include reasoning content in the assistant message.",
    )
    parser.add_argument(
        "--reasoning-type",
        default="auto",
        choices=["auto", "reasoning_full", "reasoning_no_bbox", "reasoning_no_spatial", "reasoning_short", "none"],
        help=("Which reasoning variant to extract from qa['reasoning']. "
              "'none' ignores reasoning regardless of --has_reasoning."),
    )
    parser.add_argument(
        "--pair-scope",
        default="both",
        choices=["original", "generated", "both"],
        help="Emit only original pairs, only generated pairs, or both.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/data/thangdd_workspace/InfographicDataPaper/NarrativeInfoVQA",
        help="Root directory that contains the `qas/` folder referenced by each input line.",
    )
    parser.add_argument(
        "--generated-once-per-file",
        default="true",
        help="true/false: when including generated pairs, emit them only once per qas file (recommended).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_qas_path(record_path: str | Path, dataset_root: Path) -> Path:
    qas_path = Path(record_path)
    if not qas_path.is_absolute():
        qas_path = (dataset_root / qas_path).resolve()
    return qas_path


def extract_answer(qa: Dict[str, Any]) -> str:
    """
    Return a clean single-string answer.
    Handles:
      - normalized string (e.g., "fifty")
      - stringified list/dict (e.g., "['fifty']" or "{'text': 'fifty'}")
      - list of strings or dicts (e.g., ["fifty", "50"] or [{'text': 'fifty'}])
      - SQuAD-style answers: {'text': [...]}
    """
    def pick_first_str(x) -> str:
        if isinstance(x, str) and x.strip():
            return x.strip()
        if isinstance(x, list):
            for item in x:
                if isinstance(item, str) and item.strip():
                    return item.strip()
                if isinstance(item, dict):
                    t = item.get("text")
                    if isinstance(t, str) and t.strip():
                        return t.strip()
        if isinstance(x, dict):
            t = x.get("text")
            if isinstance(t, str) and t.strip():
                return t.strip()
            if isinstance(t, list):
                for s in t:
                    if isinstance(s, str) and s.strip():
                        return s.strip()
        return ""

    # 1) Prefer 'answer' if present
    ans = qa.get("answer")

    # If it's a string, it might be a plain answer OR a stringified list/dict
    if isinstance(ans, str):
        s = ans.strip()
        if s:
            if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
                try:
                    parsed = ast.literal_eval(s)
                    pick = pick_first_str(parsed)
                    if pick:
                        return pick
                except Exception:
                    return s  # fall back to raw string
            else:
                return s

    # If it's a list/dict, normalize
    pick = pick_first_str(ans)
    if pick:
        return pick

    # 2) Fall back to SQuAD-style 'answers': {'text': [...]}
    answers = qa.get("answers")
    pick = pick_first_str(answers)
    if pick:
        return pick

    return ""


def extract_reasoning(qa: Dict[str, Any], reasoning_type: str) -> str:
    if reasoning_type == "none":
        return ""

    reasoning = qa.get("reasoning")
    if not isinstance(reasoning, dict):
        return ""

    # Explicit variant requested
    if reasoning_type in {"reasoning_full", "reasoning_no_bbox", "reasoning_no_spatial", "reasoning_short"}:
        val = reasoning.get(reasoning_type, "")
        return val.strip() if isinstance(val, str) else ""

    # AUTO mode
    merged = reasoning.get("merged_reasoning")
    if isinstance(merged, str) and merged.strip():
        return merged.strip()

    generated = reasoning.get("generated_reasoning")
    if isinstance(generated, str) and generated.strip():
        return generated.strip()

    if isinstance(generated, dict):
        for block_key in ("think", "understand"):
            block = generated.get(block_key)
            if isinstance(block, dict):
                logical = block.get("logical_reasoning")
                if isinstance(logical, str) and logical.strip():
                    return logical.strip()
                analysis = block.get("analysis")
                if isinstance(analysis, str) and analysis.strip():
                    return analysis.strip()
        answer_text = generated.get("answer")
        if isinstance(answer_text, str) and answer_text.strip():
            return answer_text.strip()

    for k in ("reasoning_short", "reasoning_full", "reasoning_no_bbox", "reasoning_no_spatial"):
        v = reasoning.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def build_conversation(
    question: str,
    reasoning: str,
    answer: str,
    has_reasoning: bool,
) -> List[Dict[str, str]]:
    # Always use the VQA prompt (minimal, with unanswerable rule)
    question_text = (
        "Answer the question according to the image using a single word or phrase. "
        "If the image does not contain enough evidence, answer exactly: unanswerable. "
        "Do not use outside knowledge.\n"
        f"Question: {(question or '').strip()}\n"
        'The last line of your response should be of the form "ANSWER: $ANSWER" '
        '(without quotes) where $ANSWER is the answer to the question.'
    )

    reasoning_text = (reasoning or "").strip()
    answer_text = (answer or "").strip()

    human = {"from": "human", "value": f"<image>{question_text}"}
    if has_reasoning and reasoning_text:
        gpt_value = f"<think>{reasoning_text}</think>ANSWER: {answer_text}"
    else:
        gpt_value = f"ANSWER: {answer_text}"
    gpt = {"from": "gpt", "value": gpt_value}

    return [human, gpt]


def sample_id(record: Dict[str, Any], qa: Dict[str, Any], pair_type: str, pair_index: int) -> str:
    for key in ("id", "qa_id", "squad_id"):
        value = qa.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    base_id = record.get("id")
    if isinstance(base_id, str) and base_id.strip():
        return f"{base_id.strip()}_{pair_type}_{pair_index}"
    return f"{pair_type}_{pair_index}"


def emit_original_only(
    record: Dict[str, Any],
    qas_data: Dict[str, Any],
    image_path: str,
    has_reasoning: bool,
    reasoning_type: str,
) -> List[Dict[str, Any]]:
    qa_type = record.get("qa_type", "original")
    qa_index = int(record.get("qa_index", 0))
    if qa_type != "original":
        return []
    pairs = qas_data.get(PAIR_KEY_BY_TYPE["original"]) or []
    if not isinstance(pairs, list) or qa_index < 0 or qa_index >= len(pairs):
        return []
    qa = pairs[qa_index]
    return [{
        "id": sample_id(record, qa, "original", qa_index),
        "image": image_path,
        "conversations": build_conversation(
            qa.get("question", ""),
            extract_reasoning(qa, reasoning_type),
            extract_answer(qa),
            has_reasoning and reasoning_type != "none",
        ),
    }]


def emit_generated_pairs(
    record: Dict[str, Any],
    qas_path: Path,
    qas_data: Dict[str, Any],
    image_path: str,
    has_reasoning: bool,
    reasoning_type: str,
    expanded_generated_files: Set[Path],
    generated_once: bool,
) -> List[Dict[str, Any]]:
    # If we only want to emit once per file and we've already done it, skip
    if generated_once and qas_path in expanded_generated_files:
        return []

    gen_pairs = qas_data.get(PAIR_KEY_BY_TYPE["generated"]) or []
    outputs: List[Dict[str, Any]] = []
    if isinstance(gen_pairs, list) and gen_pairs:
        for idx, qa in enumerate(gen_pairs):
            outputs.append({
                "id": sample_id(record, qa, "generated", idx),
                "image": image_path,
                "conversations": build_conversation(
                    qa.get("question", ""),
                    extract_reasoning(qa, reasoning_type),
                    extract_answer(qa),
                    has_reasoning and reasoning_type != "none",
                ),
            })

        # Only mark as expanded if we actually emitted something
        if generated_once:
            expanded_generated_files.add(qas_path)

    return outputs

def process_record(
    record: Dict[str, Any],
    dataset_root: Path,
    cache: Dict[Path, Dict[str, Any]],
    expanded_generated_files: Set[Path],
    has_reasoning: bool,
    reasoning_type: str,
    pair_scope: str,
    generated_once: bool
) -> List[Dict[str, Any]]:
    required_keys = ("id", "image_id", "qas_file")
    missing = [key for key in required_keys if key not in record]
    if missing:
        raise KeyError(f"Input record missing required keys: {missing}")

    qas_path = resolve_qas_path(record["qas_file"], dataset_root)
    qas_data = cache.setdefault(qas_path, load_json(qas_path))
    image_path = f"{str(record['image_id'])}.png"

    # Build each part independently
    outs_original = emit_original_only(
        record, qas_data, image_path, has_reasoning, reasoning_type
    ) if pair_scope in ("original", "both") else []

    outs_generated = (
        emit_generated_pairs(
            record, qas_path, qas_data, image_path, has_reasoning, reasoning_type,
            expanded_generated_files, generated_once
        )
        if pair_scope in ("generated", "both") else []
    )

    # Merge per requested scope
    if pair_scope == "original":
        return outs_original
    if pair_scope == "generated":
        return outs_generated
    # both
    return outs_original + outs_generated


def infer_split_from_path(p: Path) -> str:
    name = p.name.lower()
    if "train" in name:
        return "train"
    if "val" in name or "valid" in name or "dev" in name:
        return "val"
    if "test" in name:
        return "test"
    return "train"


def normalize_reasoning_suffix(has_reasoning: bool, reasoning_type: str) -> str:
    """
    Build a compact, explicit suffix for reasoning flags.
    Examples:
      has_reasoning=False or reasoning_type=='none'  -> 'noreasoning_none'
      has_reasoning=True, reasoning_type='auto'      -> 'reasoning_auto'
      has_reasoning=True, reasoning_type='reasoning_full'        -> 'reasoning_full'
      has_reasoning=True, reasoning_type='reasoning_no_bbox'     -> 'reasoning_no_bbox'
      has_reasoning=True, reasoning_type='reasoning_no_spatial'  -> 'reasoning_no_spatial'
      has_reasoning=True, reasoning_type='reasoning_short'       -> 'reasoning_short'
    """
    if (not has_reasoning) or (reasoning_type == "none"):
        return "noreasoning_none"
    # keep the explicit key for clarity/grep-ability
    return f"reasoning_{reasoning_type.replace('reasoning_', '') or 'auto'}"


def infer_split_from_path(p: Path) -> str:
    name = p.name.lower()
    if "train" in name:
        return "train"
    if "val" in name or "valid" in name or "dev" in name:
        return "val"
    if "test" in name:
        return "test"
    return "train"


def auto_output_path(input_path: Path, pair_scope: str, has_reasoning: bool, reasoning_type: str) -> Path:
    split = infer_split_from_path(input_path)
    reasoning_suffix = normalize_reasoning_suffix(has_reasoning, reasoning_type)
    filename = f"narrativeinfovqa_{split}_{pair_scope}_{reasoning_suffix}.jsonl"
    return Path(filename).resolve()



def run() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    generated_once = args.generated_once_per_file.lower() == "true"
    has_reasoning = args.has_reasoning.lower() == "true"
    reasoning_type = args.reasoning_type
    pair_scope = args.pair_scope  # original / generated / both

    # If no explicit output path, build narrativeinfovqa_{split}_{scope}_{reasoning}.jsonl
    if output_path is None:
        output_path = auto_output_path(input_path, pair_scope, has_reasoning, reasoning_type)

    cache: Dict[Path, Dict[str, Any]] = {}
    expanded_generated_files: Set[Path] = set()
    total = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            content = line.strip()
            if not content:
                continue
            try:
                record = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

            samples = process_record(
                record,
                dataset_root=dataset_root,
                cache=cache,
                expanded_generated_files=expanded_generated_files,
                has_reasoning=has_reasoning,
                reasoning_type=reasoning_type,
                pair_scope=pair_scope,
                generated_once=generated_once,
            )
            for sample in samples:
                dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += 1

    print(f"Wrote {total} samples to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    run()
