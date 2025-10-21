from typing import List, Dict, Any
from transformers import AutoProcessor
from PIL import Image
from pathlib import Path

def _find_last_sublist(lst, sub):
    if not sub or len(sub) > len(lst):
        return None
    for i in range(len(lst) - len(sub), -1, -1):
        if lst[i:i+len(sub)] == sub:
            return i
    return None

def qwen_collate(examples: List[Dict[str, Any]], processor: AutoProcessor, response_template: str):
    texts = []
    batch_images = []  # list of list-of-PIL (one list per sample)

    # Build text strings via chat template, and open images
    for ex in examples:
        messages = ex["messages"]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

        # Each sample has exactly one image in this simple case
        # (Keep a list for future multi-image compatibility)
        img_path = messages[1]["content"][0]["image"]
        img = Image.open(img_path).convert("RGB")
        batch_images.append([img])

    # Tokenize text + images together and pad
    model_inputs = processor(
        text=texts,
        images=batch_images,
        return_tensors="pt",
        padding=True
    )

    # Build labels: mask everything before and including the assistant header
    resp_ids = processor.tokenizer.encode(response_template, add_special_tokens=False)
    input_ids = model_inputs["input_ids"]
    labels = input_ids.clone()

    for i in range(input_ids.size(0)):
        ids = input_ids[i].tolist()
        start = _find_last_sublist(ids, resp_ids)
        if start is None:
            # fallback: don't train on anything (avoid corrupting weights)
            labels[i, :] = -100
        else:
            cutoff = start + len(resp_ids)
            labels[i, :cutoff] = -100

    model_inputs["labels"] = labels
    return model_inputs
