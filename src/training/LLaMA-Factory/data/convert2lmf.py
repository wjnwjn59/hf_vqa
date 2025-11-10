import argparse
import json
import os
from pathlib import Path
from typing import Iterable


DATA_CONTAINER_KEYS = ("data", "items", "annotations", "records")


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_entries(path: Path) -> Iterable[dict]:
    """
    Yield dataset entries regardless of whether the source file is JSON or JSONL.
    JSON files can optionally wrap the entries inside keys such as `data`.
    """
    try:
        with path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)
    except json.JSONDecodeError:
        # Fall back to JSONL streaming when a standard JSON decode fails.
        yield from _iter_jsonl(path)
        return

    if isinstance(payload, list):
        yield from payload
        return

    if isinstance(payload, dict):
        for key in DATA_CONTAINER_KEYS:
            if isinstance(payload.get(key), list):
                yield from payload[key]
                return
        # Treat the entire dictionary as a single entry when no list wrapper exists.
        yield payload
        return

    raise ValueError(f"Unsupported JSON payload structure in {path}")


def format_entry(raw: dict, image_col: str, question_id_col: str, answer_col: str, image_dir: Path, use_vqa_prompt: bool) -> dict:
    image_name = raw.get(image_col)
    
    image_path = os.path.join(str(image_dir), image_name) if image_name else None
    entry = {
        "image": image_path,
        "question": raw.get("question"),
        "question_id": raw.get(question_id_col),
        "answer": raw.get(answer_col, []),
        # "answer_type": raw.get("answer_type", []),
        # "evidence": raw.get("evidence", []),
        # "operation/reasoning": raw.get("operation/reasoning", []),
        # "ocr_output_file": raw.get("ocr_output_file"),
        "predict": raw.get("predict", ""),
    }
    

    answer_for_conversation = entry["answer"]
    if isinstance(answer_for_conversation, (dict, list)):
        answer_text = json.dumps(answer_for_conversation, ensure_ascii=False)
    else:
        answer_text = "" if answer_for_conversation is None else str(answer_for_conversation)

    if use_vqa_prompt:
        prompt = f"Answer the question according to the image using a single word or phrase.\n{entry['question']}\nThe last line of your response should be of the form \"ANSWER: $ANSWER\" (without quotes) where $ANSWER is the answer to the question."
        entry["conversations"] = [
            {"from": "human", "value": f"<image>{prompt}"},
            {"from": "gpt", "value": answer_text},
        ]
    else:
        entry["conversations"] = [
            {"from": "human", "value": f"<image>{entry['question']}"},
            {"from": "gpt", "value": answer_text},
        ]
    return entry


def dump_jsonl(entries, path: Path):
    with path.open("w", encoding="utf-8") as outfile:
        for entry in entries:
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")


def convert_file(input_path: Path, output_path: Path, image_col: str , question_id_col: str, answer_col:str, image_dir: Path, use_vqa_prompt: bool):
    formatted_entries = (format_entry(raw, image_col, question_id_col, answer_col, image_dir, use_vqa_prompt) for raw in load_entries(input_path))
    dump_jsonl(formatted_entries, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert InfographicVQA JSONL to chat-friendly JSONL.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        )
    parser.add_argument(
        "--vqa_prompt",
        type=bool,
        default=True,
        )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    use_vqa_prompt = args.vqa_prompt
    
    input, output, image_col, question_id_col, answer_col, image_dir = None, None, None, None, None, None
    if dataset == "infographicvqa":
        input = Path("/mnt/VLAI_data/InfographicVQA/question_answer/infographicsVQA_val_v1.0_withQT.json")
        output = Path("data/infographicvqa_val_lmf.jsonl")
        image_dir = Path("/mnt/VLAI_data/InfographicVQA/images")
        image_col = "image_local_name"
        question_id_col = "questionId"
        answer_col = "answers"
    elif dataset == "infographicvqa_test":
        input = Path("/mnt/VLAI_data/InfographicVQA/question_answer/infographicsVQA_test_v1.0.json")
        output = Path("data/infographicvqa_test_lmf.jsonl")
        image_dir = Path("/mnt/VLAI_data/InfographicVQA/images")
        image_col = "image_local_name"
        question_id_col = "questionId"
        answer_col = ""
    elif dataset == "docvqa":
        input = Path("/mnt/VLAI_data/DocVQA/val.jsonl")
        output = Path("data/docvqa_val_lmf.jsonl")
        image_dir = Path("/mnt/VLAI_data/DocVQA/images")
        image_col = "image"
        question_id_col = "question_id"
        answer_col = "answer"
    elif dataset == "textvqa":
        input = Path("/mnt/VLAI_data/TextVQA/textvqa_val_updated.jsonl")
        output = Path("data/textvqa_val_lmf.jsonl")
        image_dir = Path("/mnt/VLAI_data/TextVQA/train_images")
        image_col = "image"
        question_id_col = "question_id"
        answer_col = "answer"
    elif dataset == "chartqa_human":
        input = Path("/mnt/VLAI_data/ChartQA/test_human.jsonl")
        output = Path("data/chartqa_val_human_lmf.jsonl")
        image_dir = Path("/mnt/VLAI_data/ChartQA/test")
        image_col = "image"
        question_id_col = "question_id"
        answer_col = "answer"
    elif dataset == "chartqa_aug":
        input = Path("/mnt/VLAI_data/ChartQA/test_augmented.jsonl")
        output = Path("data/chartqa_val_aug_lmf.jsonl")
        image_dir = Path("/mnt/VLAI_data/ChartQA/test")
        image_col = "image"
        question_id_col = "question_id"
        answer_col = "answer"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
        
    print("-"*64)
    print(f"Converting {input}\nto {output}...")
    print("-"*64)
    convert_file(input, output, image_col, question_id_col, answer_col, image_dir, use_vqa_prompt)


if __name__ == "__main__":
    main()
