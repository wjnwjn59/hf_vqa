# pip install datasets jinja2 transformers accelerate torch regex --upgrade
# Optional (recommended) for sentence splitting:
# pip install blingfire
# OR: pip install nltk

import os, re, json, argparse, time
from typing import List, Dict, Any, Optional
import torch
from jinja2 import Template
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ======================
# Global settings
# ======================
MODEL_PATH = "/data/thangdd_workspace/InfographicDataPaper/llm_checkpoints/Qwen_Qwen3-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

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
# Model loading / generation
# ======================

def load_model_and_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    # Set a safe pad_token_id to avoid generation warnings/errors
    if getattr(model, "generation_config", None) is not None:
        if model.generation_config.pad_token_id is None and tok.eos_token_id is not None:
            model.generation_config.pad_token_id = tok.eos_token_id
    return tok, model

def is_chat_template(tokenizer: AutoTokenizer) -> bool:
    return hasattr(tokenizer, "apply_chat_template")

def run_llm(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    *,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    system: Optional[str] = None
) -> str:
    use_chat = hasattr(tokenizer, "apply_chat_template")

    if use_chat:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        input_len = input_ids.shape[-1]

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=(model.generation_config.pad_token_id
                              if getattr(model, "generation_config", None) is not None
                              else tokenizer.eos_token_id),
            )
        gen = out[0][input_len:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    else:
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=(model.generation_config.pad_token_id
                              if getattr(model, "generation_config", None) is not None
                              else tokenizer.eos_token_id),
            )
        gen = out[0][enc.input_ids.shape[-1]:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

def postprocess_chat_output(text: str) -> str:
    return text.strip()

# ======================
# JSON extraction (balanced-brace scan; no recursive regex)
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
# Sentence split (BlingFire -> NLTK -> regex fallback)
# ======================

def split_into_sentences(context: str, mode: str = "auto") -> List[str]:
    """
    mode:
      - 'auto'  : try blingfire -> nltk -> fallback regex
      - 'bling' : force blingfire
      - 'nltk'  : force nltk
      - 'regex' : force fallback regex
    """
    ctx = " ".join((context or "").strip().split())
    if not ctx:
        return []

    if mode in ("auto", "bling"):
        try:
            from blingfire import text_to_sentences
            s = text_to_sentences(ctx).strip()
            out = [line.strip() for line in s.split("\n") if line.strip()]
            if out or mode == "bling":
                return out
        except Exception:
            if mode == "bling":
                return [ctx]
            pass

    if mode in ("auto", "nltk"):
        try:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            from nltk.tokenize import sent_tokenize
            out = [s.strip() for s in sent_tokenize(ctx) if s.strip()]
            if out or mode == "nltk":
                return out
        except Exception:
            if mode == "nltk":
                return [ctx]
            pass

    # Regex fallback (protect common abbreviations and decimals)
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
# Parsers for Stage 1/2
# ======================

def parse_stage_a(text: str) -> List[Dict[str, Any]]:
    js = extract_first_json(text)
    if not js or "summaries" not in js:
        raise ValueError("Stage 1: output not parseable as expected JSON with key 'summaries'.")
    items = js["summaries"]
    for it in items:
        if "id" not in it or "summary" not in it:
            raise ValueError("Stage 1: item missing 'id' or 'summary'.")
        it["id"] = int(it["id"])
        it["summary"] = str(it["summary"]).strip()
    items.sort(key=lambda x: x["id"])
    return items

def parse_stage_b(text: str) -> List[Dict[str, Any]]:
    js = extract_first_json(text)
    if not js or "figures" not in js:
        raise ValueError("Stage 2: output not parseable as expected JSON with key 'figures'.")
    items = js["figures"]
    for it in items:
        if "id" not in it or "ideas" not in it:
            raise ValueError("Stage 2: item missing 'id' or 'ideas'.")
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
# Pipeline stages
# ======================

def stage_a_summarize(tokenizer, model, sents_enum, stage_a_tmpl_text: str):
    prompt = render_template(stage_a_tmpl_text, sents=sents_enum)
    raw = run_llm(
        tokenizer, model, prompt,
        temperature=0.2, top_p=0.9, max_new_tokens=1200,
        system="Return strictly valid JSON for the task."
    )
    return parse_stage_a(raw)

def stage_b_figures(tokenizer, model, items, stage_b_tmpl_text: str):
    prompt = render_template(stage_b_tmpl_text, items=items)
    raw = run_llm(
        tokenizer, model, prompt,
        temperature=0.5, top_p=0.9, max_new_tokens=1200,
        system="Return strictly valid JSON for the task."
    )
    return parse_stage_b(raw)

def stage_c_compose(tokenizer, model, items, stage_c_tmpl_text: str) -> str:
    prompt = render_template(stage_c_tmpl_text, items=items)
    raw = run_llm(
        tokenizer, model, prompt,
        temperature=0.4, top_p=0.9, max_new_tokens=1200,
        system="Write a single-paragraph infographic design brief."
    )
    final_str = " ".join(raw.strip().split())
    return final_str

# ======================
# End-to-end per sample (with saving)
# ======================

def process_sample(
    tokenizer, model,
    context: str,
    stage_a_path: str,
    stage_b_path: str,
    stage_c_path: str,
    split_mode: str,
    out_dir: str,
    sample_name: str
) -> Dict[str, Any]:

    # Ensure template files exist
    ensure_file(stage_a_path, "Stage 1")
    ensure_file(stage_b_path, "Stage 2")
    ensure_file(stage_c_path, "Stage 3")

    stage_a_tmpl_text = read_text(stage_a_path)
    stage_b_tmpl_text = read_text(stage_b_path)
    stage_c_tmpl_text = read_text(stage_c_path)

    # Create sample dir
    sample_dir = os.path.join(out_dir, sample_name)
    ensure_dir(sample_dir)

    # Stage 0: sentence segmentation
    sents = split_into_sentences(context, mode=split_mode)
    sents_enum = enumerate_sentences(sents)
    save_json(os.path.join(sample_dir, "sentences.json"), {"sentences": sents_enum})

    # Stage 1: summaries
    summaries = stage_a_summarize(tokenizer, model, sents_enum, stage_a_tmpl_text)
    save_json(os.path.join(sample_dir, "summaries.json"), {"summaries": summaries})

    # Stage 2: figures
    figures = stage_b_figures(tokenizer, model, summaries, stage_b_tmpl_text)
    save_json(os.path.join(sample_dir, "figures.json"), {"figures": figures})

    # Merge 1 + 2 by id
    merged = {it["id"]: {"id": it["id"], "summary": it["summary"], "ideas": []} for it in summaries}
    for it in figures:
        if it["id"] in merged:
            merged[it["id"]]["ideas"] = it["ideas"]
    merged_items = [merged[k] for k in sorted(merged.keys())]

    # Stage 3: final caption
    final_desc = stage_c_compose(tokenizer, model, merged_items, stage_c_tmpl_text)
    save_json(os.path.join(sample_dir, "final_description.json"), {"final_description": final_desc})

    # Combined
    combined = {
        "sentences": sents_enum,
        "summaries": summaries,
        "figures": merged_items,
        "final_description": final_desc
    }
    save_json(os.path.join(sample_dir, "combined.json"), combined)

    return combined

# ======================
# CLI
# ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_a",
                        default="hf_vqa/src/prompts/content_des_stage_1.jinja",
                        help="Path to Stage 1 Jinja (.jinja)")
    parser.add_argument("--stage_b",
                        default="hf_vqa/src/prompts/content_des_stage_2.jinja",
                        help="Path to Stage 2 Jinja (.jinja)")
    parser.add_argument("--stage_c",
                        default="hf_vqa/src/prompts/content_des_stage_3.jinja",
                        help="Path to Stage 3 Jinja (.jinja)")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--context", type=str, default=None,
                        help="Override with a custom context string")
    parser.add_argument("--split_mode", type=str, default="auto",
                        choices=["auto", "bling", "nltk", "regex"],
                        help="Sentence splitting backend.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to write JSON outputs.")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Folder name prefix for saved samples.")
    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer(MODEL_PATH)

    ensure_dir(args.output_dir)

    if args.context:
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = args.prefix or f"custom_{ts}"
        result = process_sample(
            tokenizer, model, args.context,
            args.stage_a, args.stage_b, args.stage_c,
            split_mode=args.split_mode,
            out_dir=args.output_dir,
            sample_name=name
        )
        print(result["final_description"])
        return

    # Demo using HF SQuAD v2 rows (context only)
    ds = load_dataset("squad_v2", split=args.split)
    for i in range(args.n_samples):
        row = ds[i]
        context = row["context"]
        name = (args.prefix + f"_sample_{i:05d}") if args.prefix else f"sample_{i:05d}"
        result = process_sample(
            tokenizer, model, context,
            args.stage_a, args.stage_b, args.stage_c,
            split_mode=args.split_mode,
            out_dir=args.output_dir,
            sample_name=name
        )
        print("\n=== FINAL DESCRIPTION ===\n")
        print(result["final_description"])
        print("\nSaved to:", os.path.join(args.output_dir, name))
        print("\n=========================\n")

if __name__ == "__main__":
    main()
