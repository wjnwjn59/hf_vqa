from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

PAIR_KEY_BY_TYPE = {
    "original": "original_qa_pairs",
    "generated": "generated_qa_pairs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert QA JSON files into conversation JSONL rows.")
    parser.add_argument("--input", help="Path to the input JSONL definition file.", default="/home/thinhnp/hf_vqa/dataset/train_annotations.jsonl")
    parser.add_argument("--output", help="Path to the output JSONL that will be created.", default="data/train_sft.jsonl")
    parser.add_argument("--has_reasoning", required=True, help="If set, include reasoning in the gpt response.")
    parser.add_argument("--has_vqa_prompt", required=True, help="If set, include VQA prompt in the human question.")
    parser.add_argument(
        "--dataset-root",
        default="/home/thinhnp/hf_vqa/dataset/",
        help="Directory that contains the `qas/` folder referenced by each input line (default: current directory).",
    )
    parser.add_argument(
        "--skip-generated",
        action="store_true",
        help="If set, do not automatically expand `generated_qa_pairs`. Only the referenced QA pair is emitted.",
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
    answer = qa.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    answers = qa.get("answers")
    if isinstance(answers, dict):
        texts = answers.get("text")
        if isinstance(texts, list) and texts:
            candidate = texts[0]
            if isinstance(candidate, str):
                return candidate.strip()

    # Some generated pairs store a list of answers directly under `answer`.
    if isinstance(answer, list) and answer:
        candidate = answer[0]
        if isinstance(candidate, str):
            return candidate.strip()

    return ""


def extract_reasoning(qa: Dict[str, Any]) -> str:
    reasoning = qa.get("reasoning")
    if not isinstance(reasoning, dict):
        return ""
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
    return ""


def build_conversation(question: str, reasoning: str, answer: str, has_reasoning: bool, has_vqa_promt: bool) -> List[Dict[str, str]]:
    if has_vqa_promt:
        question_text = f"""Answer the question according to the image using a single word or phrase.
        {(question or "").strip()}
        The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the question.""".strip(),
    else:
        question_text = (question or "").strip()
    
    reasoning_text = (reasoning or "").strip()
    answer_text = (answer or "").strip()
    human = {"from": "human", "value": f"<image>{question_text}"}
    
    if has_reasoning:
        gpt_value = f"<think>{reasoning_text}</think><answer>{answer_text}</answer>"
    else:
        gpt_value = f"{answer_text}"
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


def iter_pairs(
    qas_data: Dict[str, Any],
    target_type: str,
    target_index: int,
    include_generated: bool,
) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    # Always yield the explicitly referenced pair first.
    pair_key = PAIR_KEY_BY_TYPE.get(target_type)
    if not pair_key:
        raise ValueError(f"Unsupported qa_type '{target_type}'. Expected one of {sorted(PAIR_KEY_BY_TYPE)}")
    pairs = qas_data.get(pair_key) or []
    if not isinstance(pairs, list) or target_index >= len(pairs) or target_index < 0:
        raise IndexError(f"qa_index {target_index} is invalid for '{target_type}' (found {len(pairs)} pairs)")
    yielded = {(target_type, target_index)}
    yield target_type, target_index, pairs[target_index]

    # Optionally expand every generated QA once per file.
    if not include_generated:
        return
    generated_pairs = qas_data.get("generated_qa_pairs") or []
    if not isinstance(generated_pairs, list):
        return
    for idx, qa in enumerate(generated_pairs):
        if ("generated", idx) in yielded:
            continue
        yield "generated", idx, qa


def process_record(
    record: Dict[str, Any],
    dataset_root: Path,
    include_generated: bool,
    cache: Dict[Path, Dict[str, Any]],
    generated_emitted: Set[Path],
    has_reasoning: bool,
    has_vqa_promt: bool,
) -> List[Dict[str, Any]]:
    required_keys = ("id", "image_id", "qas_file")
    missing = [key for key in required_keys if key not in record]
    if missing:
        raise KeyError(f"Input record missing required keys: {missing}")

    qas_path = resolve_qas_path(record["qas_file"], dataset_root)
    qas_data = cache.setdefault(qas_path, load_json(qas_path))

    include_file_generated = include_generated and qas_path not in generated_emitted

    qa_type = record.get("qa_type", "original")
    qa_index = int(record.get("qa_index", 0))

    image_path = f"{str(record['image_id'])}.png"
    outputs: List[Dict[str, Any]] = []

    for pair_type, pair_idx, qa in iter_pairs(qas_data, qa_type, qa_index, include_file_generated):
        sample = {
            "id": sample_id(record, qa, pair_type, pair_idx),
            "image": image_path,
            "conversations": build_conversation(
                qa.get("question", ""),
                extract_reasoning(qa),
                extract_answer(qa),
                has_reasoning,
                has_vqa_promt,
            ),
        }
        outputs.append(sample)

    if include_file_generated and qas_path not in generated_emitted:
        generated_emitted.add(qas_path)

    return outputs


def run() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    input_path = Path(args.input)
    output_path = Path(args.output)
    has_reasoning = args.has_reasoning.lower() == "true"
    has_vqa_prompt = args.has_vqa_prompt.lower() == "true"
    
    if has_reasoning:
        output_path = output_path.with_name(output_path.stem + "_reasoning" + output_path.suffix)
    if has_vqa_prompt:
        output_path = output_path.with_name(output_path.stem + "_vqaprompt" + output_path.suffix)

    cache: Dict[Path, Dict[str, Any]] = {}
    generated_emitted: Set[Path] = set()
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
                include_generated=not args.skip_generated,
                cache=cache,
                generated_emitted=generated_emitted,
                has_reasoning=has_reasoning,
                has_vqa_promt=has_vqa_prompt,
            )
            for sample in samples:
                dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += 1

    print(f"Wrote {total} samples to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    run()
