from typing import List, Dict, Any, Tuple, Optional, Protocol
import json
import os
import argparse
import random
import re
import sys
from pathlib import Path
from jinja2 import Template
from tqdm import tqdm

# Import the Qwen3 inference module
from src.inference.qwen3_inference import Qwen3Inference


# ============================================================================
# LLM PROTOCOL AND BACKENDS
# ============================================================================

class LLM(Protocol):
    def generate_single(self, prompt: str, enable_thinking: bool = False) -> str: ...


class ChatGPTInference:
    """
    Adapter that matches Qwen3Inference.generate_single interface
    but uses the OpenAI Responses API under the hood.
    """
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        # Lazy import keeps dependency optional unless backend=gpt is chosen
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.system_prompt = system_prompt

    def generate_single(self, prompt: str, enable_thinking: bool = False) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.responses.create(
            model=self.model_name,
            input=messages,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        return resp.output_text.strip()


# ============================================================================
# I/O HELPERS AND UTILITIES
# ============================================================================

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


# ============================================================================
# TEXT PROCESSING AND PARSING
# ============================================================================

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
            c = text[i]
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = not in_str
            elif not in_str:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except:
                            break
            i += 1
        start = text.find('{', start + 1)
    return None


# ============================================================================
# STAGE PARSERS
# ============================================================================

def parse_stage_1(text: str) -> Dict[str, Any]:
    """Parse Stage 1 response to extract segments and figures"""
    js = extract_first_json(text)
    if not js:
        raise ValueError(f"No valid JSON found in Stage 1 response: {text[:200]}...")

    if "segments" not in js or "figures" not in js:
        raise ValueError(f"Missing 'segments' or 'figures' in Stage 1 response: {js}")

    # Validate segments
    segments = js["segments"]
    for segment in segments:
        if "id" not in segment or "text" not in segment:
            raise ValueError(f"Invalid segment format: {segment}")
        segment["id"] = int(segment["id"])

    # Validate figures
    figures = js["figures"]
    for figure in figures:
        if "id" not in figure or "description" not in figure:
            raise ValueError(f"Invalid figure format: {figure}")
        figure["id"] = int(figure["id"])

    # Sort by id
    segments.sort(key=lambda x: x["id"])
    figures.sort(key=lambda x: x["id"])

    return {
        'title': js.get('title', ''),
        'segments': segments,
        'figures': figures
    }


# ============================================================================
# KEYWORD CHECKING UTILITIES
# ============================================================================

def extract_answer_keywords(qa_pairs: List[Dict]) -> List[str]:
    """Extract all answer keywords from QA pairs, excluding impossible questions and short/common words"""
    excluded_words = {
        'yes', 'no', 'true', 'false', 'a', 'an', 'the', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
        'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'why', 'how',
        'and', 'or', 'but', 'so', 'yet', 'for', 'nor', 'as', 'if', 'then', 'else',
        'in', 'on', 'at', 'by', 'to', 'of', 'from', 'up', 'down', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'very', 'too', 'also', 'just', 'only'
    }

    keywords = []
    for qa in qa_pairs:
        answers = qa.get('answers', {})

        if not answers or not answers.get('text') or len(answers.get('text', [])) == 0:
            continue

        if 'text' in answers and isinstance(answers['text'], list):
            valid_answers = [ans.strip() for ans in answers['text'] if ans.strip()]

            filtered_answers = []
            for answer in valid_answers:
                answer_lower = answer.lower().strip()

                if len(answer_lower) <= 2:
                    continue
                if answer_lower in excluded_words:
                    continue
                if answer_lower.isdigit():
                    continue
                if len(answer_lower) == 4 and answer_lower.isdigit():
                    continue
                if len(answer_lower) <= 3 and answer_lower in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use', 'six'}:
                    continue

                if len(answer_lower) >= 4 or answer.isupper() or any(char.isupper() for char in answer):
                    filtered_answers.append(answer)
                elif len(answer.split()) > 1:
                    filtered_answers.append(answer)

            keywords.extend(filtered_answers)

    unique_keywords = []
    seen = set()
    for kw in keywords:
        if kw.lower() not in seen:
            unique_keywords.append(kw)
            seen.add(kw.lower())

    return unique_keywords

def has_answerable_questions(qa_pairs: List[Dict]) -> bool:
    """Check if there are any answerable questions in the QA pairs"""
    for qa in qa_pairs:
        answers = qa.get('answers', {})

        if answers and answers.get('text') and len(answers.get('text', [])) > 0:
            valid_answers = [ans.strip() for ans in answers['text'] if ans.strip()]
            if valid_answers:
                return True
    return False

def check_keywords_in_caption(caption: str, keywords: List[str]) -> tuple[bool, List[str]]:
    """Check if any of the keywords appear in the caption (case insensitive)"""
    if not caption or not keywords:
        return False, []

    caption_lower = caption.lower()
    found_keywords = []

    for keyword in keywords:
        keyword_lower = keyword.lower()

        if keyword_lower in caption_lower:
            found_keywords.append(keyword)
            continue

        if ' ' in keyword_lower:
            words = keyword_lower.split()
            for word in words:
                if len(word) > 3 and word in caption_lower:
                    found_keywords.append(keyword)
                    break
        else:
            if len(keyword_lower) > 4:
                words_in_caption = caption_lower.split()
                for word in words_in_caption:
                    if keyword_lower in word or word in keyword_lower:
                        found_keywords.append(keyword)
                        break

    return len(found_keywords) > 0, found_keywords


# ============================================================================
# STAGE PROCESSING FUNCTIONS
# ============================================================================

def stage_1_generate_content(llm: LLM, context: str, qa_pairs: List[Dict], stage_1_tmpl_text: str) -> Dict[str, Any]:
    """
    Stage 1: Generate segments and figures from context and QA pairs
    """
    qa_samples = []
    if qa_pairs:
        for i, qa in enumerate(qa_pairs):
            answer_text = ""
            if 'answers' in qa and qa['answers']:
                if isinstance(qa['answers'], dict):
                    if 'text' in qa['answers']:
                        if isinstance(qa['answers']['text'], list) and qa['answers']['text']:
                            answer_text = qa['answers']['text'][0]
                        else:
                            answer_text = str(qa['answers']['text'])
                    elif 'answer_start' in qa['answers'] and 'text' in qa['answers']:
                        if isinstance(qa['answers']['text'], list) and qa['answers']['text']:
                            answer_text = qa['answers']['text'][0]
                elif isinstance(qa['answers'], list) and qa['answers']:
                    answer_text = str(qa['answers'][0])
                else:
                    answer_text = str(qa['answers'])

            qa_samples.append({
                'id': i + 1,
                'question': qa.get('question', ''),
                'answer': answer_text
            })

    prompt = render_template(stage_1_tmpl_text, context=context, qa_samples=qa_samples)
    response = llm.generate_single(
        prompt,
        enable_thinking=False
    )
    return parse_stage_1(response)

def stage_2_generate_caption(llm: LLM, layout_elements: List[Dict], stage_2_tmpl_text: str, canvas_width: int = 896, canvas_height: int = 2240) -> str:
    """
    Stage 2: Generate full image caption from layout elements
    """
    prompt = render_template(
        stage_2_tmpl_text,
        layout_elements=layout_elements,
        canvas_width=canvas_width,
        canvas_height=canvas_height
    )
    response = llm.generate_single(
        prompt,
        enable_thinking=False
    )
    return response.strip()


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_squad_v2_data(input_path: str) -> List[Dict[str, Any]]:
    """Load Squad v2 data from JSONL file with context deduplication, keeping all QA pairs for each unique context"""
    all_data = []
    seen_contexts = {}
    total_entries = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            total_entries += 1

            context = entry.get('context', '').strip()
            if not context:
                continue

            qa_pair = {
                'question': entry.get('question', ''),
                'answers': entry.get('answers', {}),
                'id': entry.get('id', '')
            }

            if context not in seen_contexts:
                seen_contexts[context] = {
                    'context': context,
                    'qa_pairs': []
                }
            seen_contexts[context]['qa_pairs'].append(qa_pair)

    for context_data in seen_contexts.values():
        all_data.append(context_data)

    total_qa_pairs = sum(len(item['qa_pairs']) for item in all_data)
    print(f"Loaded {total_entries} total entries from Squad v2 file: {input_path}")
    print(f"Unique contexts: {len(all_data)} (deduplication removed {total_entries - len(all_data)} entries)")
    print(f"Total QA pairs: {total_qa_pairs}")
    return all_data

def save_chunk_to_file(chunk: List[Optional[Dict]], output_dir: str, file_index: int) -> Optional[str]:
    """Save a chunk of results to file in wiki format"""
    if not chunk:
        return None

    wiki_layouts = []
    for result in chunk:
        if result is not None and result.get('success', False) and 'wiki_layout' in result:
            wiki_layouts.append(result['wiki_layout'])

    if not wiki_layouts:
        print(f"  ✗ No valid wiki layouts to save for file {file_index:06d}")
        return None

    filename = f"wiki{file_index:06d}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(wiki_layouts, f, ensure_ascii=False, indent=2)

    none_count = len(chunk) - len(wiki_layouts)
    none_info = f" ({none_count} failed keyword checks)" if none_count > 0 else ""

    print(f"  ✓ Saved {len(wiki_layouts)} wiki layouts to {filename}{none_info}")
    if wiki_layouts:
        print(f"    Wiki IDs: {wiki_layouts[0]['index']}-{wiki_layouts[-1]['index']}")

    return filename


# ============================================================================
# IMPROVED LAYOUT ALGORITHMS
# ============================================================================

def calculate_text_readability_score(text_bbox: Dict, text_content: str) -> float:
    """
    Calculate readability score for text based on area vs content length.
    """
    area = calculate_bbox_area(text_bbox)
    char_count = len(text_content)

    if char_count == 0:
        return 1.0

    min_area_per_char = 30
    ideal_area = char_count * min_area_per_char

    score = min(area / ideal_area, 1.0) if ideal_area > 0 else 0.0
    return score


def add_safe_margins(bbox: Dict, margin: int = 20, canvas_width: int = 896, canvas_height: int = 2240) -> Dict:
    """
    Add safe margins to a bounding box to avoid edge placement.
    """
    top_left = bbox['top_left']
    bottom_right = bbox['bottom_right']

    new_x = max(margin, top_left[0])
    new_y = max(margin, top_left[1])

    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]

    if new_x + width > canvas_width - margin:
        new_x = canvas_width - margin - width

    if new_y + height > canvas_height - margin:
        new_y = canvas_height - margin - height

    new_x = max(margin, new_x)
    new_y = max(margin, new_y)

    return {
        'category': bbox['category'],
        'top_left': [new_x, new_y],
        'bottom_right': [new_x + width, new_y + height],
        'caption': bbox.get('caption', ''),
        'text': bbox.get('text', '')
    }


def calculate_overlap_penalty(text_bbox: Dict, image_bboxes: List[Dict]) -> float:
    """
    Calculate overlap penalty score for a text bbox against image bboxes.
    """
    total_penalty = 0.0

    for img_bbox in image_bboxes:
        overlap_ratio = check_overlap_ratio(text_bbox, img_bbox)
        if overlap_ratio > 0:
            total_penalty += overlap_ratio * 10

    return total_penalty


def find_best_text_positions(
    text_elements: List[str],
    candidate_text_bboxes: List[Dict],
    image_bboxes: List[Dict],
    margin: int = 20
) -> List[Tuple[Dict, str]]:
    """
    Find optimal text positions to minimize overlaps and improve readability.
    """
    if not text_elements or not candidate_text_bboxes:
        return []

    bbox_scores = []
    for bbox in candidate_text_bboxes:
        safe_bbox = add_safe_margins(bbox, margin)

        overlap_penalty = calculate_overlap_penalty(safe_bbox, image_bboxes)

        area_score = calculate_bbox_area(safe_bbox) / 50000.0

        x, y = safe_bbox['top_left']
        position_score = min(x / 100.0, 1.0) * min(y / 100.0, 1.0)

        total_score = overlap_penalty - area_score - position_score

        bbox_scores.append((safe_bbox, total_score))

    bbox_scores.sort(key=lambda x: x[1])

    result_pairs = []
    used_bboxes = []

    for i, text_content in enumerate(text_elements):
        if i >= len(bbox_scores):
            break

        best_bbox = None
        best_score = float('inf')

        for bbox, score in bbox_scores:
            if bbox in used_bboxes:
                continue

            readability_score = calculate_text_readability_score(bbox, text_content)
            if readability_score < 0.5:
                score += 5.0

            text_overlap_penalty = 0.0
            for used_bbox in used_bboxes:
                if used_bbox.get('category') == 'text':
                    overlap_ratio = check_overlap_ratio(bbox, used_bbox)
                    if overlap_ratio > 0.1:
                        text_overlap_penalty += overlap_ratio * 5.0

            final_score = score + text_overlap_penalty

            if final_score < best_score:
                best_score = final_score
                best_bbox = bbox

        if best_bbox:
            best_bbox = dict(best_bbox)
            used_bboxes.append(best_bbox)
            result_pairs.append((best_bbox, text_content))

    return result_pairs


def smart_image_selection(
    image_elements: List[Dict],
    candidate_element_bboxes: List[Dict],
    reserved_text_areas: List[Dict] = None
) -> List[Tuple[Dict, str]]:
    """
    Smart selection of image positions to minimize conflicts with text areas.
    """
    if not image_elements or not candidate_element_bboxes:
        return []

    reserved_text_areas = reserved_text_areas or []

    bbox_scores = []
    for bbox in candidate_element_bboxes:
        score = 0.0

        for text_area in reserved_text_areas:
            overlap_ratio = check_overlap_ratio(bbox, text_area)
            if overlap_ratio > 0:
                score += overlap_ratio * 8.0

        area_score = calculate_bbox_area(bbox) / 100000.0
        score -= area_score

        bbox_scores.append((bbox, score))

    bbox_scores.sort(key=lambda x: x[1])

    result_pairs = []
    used_bboxes = []

    for i, image_element in enumerate(image_elements):
        if i >= len(bbox_scores):
            break

        best_bbox = None
        best_score = float('inf')

        for bbox, score in bbox_scores:
            if bbox in used_bboxes:
                continue

            image_overlap_penalty = 0.0
            for used_bbox in used_bboxes:
                overlap_ratio = check_overlap_ratio(bbox, used_bbox)
                if overlap_ratio > 0.3:
                    image_overlap_penalty += overlap_ratio * 3.0

            final_score = score + image_overlap_penalty

            if final_score < best_score:
                best_score = final_score
                best_bbox = bbox

        if best_bbox:
            used_bboxes.append(best_bbox)
            caption = image_element.get('description', 'decorative element')
            result_pairs.append((best_bbox, caption))

    return result_pairs


def validate_layout_quality(layers: List[Dict]) -> Dict:
    """
    Validate the quality of a generated layout.
    """
    text_layers = [l for l in layers if l.get('category') == 'text']
    image_layers = [l for l in layers if l.get('category') == 'element' and l.get('bottom_right') != [896, 2240]]

    issues = []

    severe_overlaps = 0
    for text_layer in text_layers:
        for image_layer in image_layers:
            overlap_ratio = check_overlap_ratio(text_layer, image_layer)
            if overlap_ratio > 0.3:
                severe_overlaps += 1
                issues.append(f"Text '{text_layer.get('text', '')[:30]}...' overlaps with image")

    readability_issues = 0
    for text_layer in text_layers:
        text_content = text_layer.get('text', '')
        readability_score = calculate_text_readability_score(text_layer, text_content)
        if readability_score < 0.4:
            readability_issues += 1
            issues.append(f"Poor readability for text '{text_content[:30]}...'")

    edge_issues = 0
    for text_layer in text_layers:
        x, y = text_layer['top_left']
        if x < 15 or y < 15:
            edge_issues += 1
            issues.append(f"Text too close to edge: {text_layer.get('text', '')[:30]}...")

    quality_score = 1.0
    if severe_overlaps > 0:
        quality_score -= 0.4
    if readability_issues > 0:
        quality_score -= 0.3
    if edge_issues > 0:
        quality_score -= 0.2

    return {
        'quality_score': max(0.0, quality_score),
        'severe_overlaps': severe_overlaps,
        'readability_issues': readability_issues,
        'edge_issues': edge_issues,
        'issues': issues,
        'passes_quality': quality_score > 0.6
    }


def check_overlap_ratio(bbox1: Dict, bbox2: Dict) -> float:
    """Calculate overlap ratio (intersection / smaller_area)."""
    x1_min, y1_min = bbox1['top_left']
    x1_max, y1_max = bbox1['bottom_right']
    x2_min, y2_min = bbox2['top_left']
    x2_max, y2_max = bbox2['bottom_right']

    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap

    if intersection == 0:
        return 0.0

    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)
    smaller_area = min(area1, area2)

    return intersection / smaller_area if smaller_area > 0 else 0.0


def auto_scale_small_text_bboxes(
    layers: List[Dict],
    canvas_width: int = 896,
    canvas_height: int = 2240,
    min_text_area: int = 2000
) -> List[Dict]:
    """
    Auto-scale small text bboxes to improve readability.
    """
    updated_layers = []

    for layer in layers:
        if layer.get('category') == 'text':
            area = calculate_bbox_area(layer)
            text_content = layer.get('text', '')

            if area < min_text_area and len(text_content) > 10:
                char_count = len(text_content)
                ideal_area = char_count * 40
                scale_factor = min(2.0, (ideal_area / area) ** 0.5) if area > 0 else 1.5

                top_left = layer['top_left']
                bottom_right = layer['bottom_right']
                width = bottom_right[0] - top_left[0]
                height = bottom_right[1] - top_left[1]

                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                if top_left[0] + new_width > canvas_width:
                    new_width = canvas_width - top_left[0] - 10
                if top_left[1] + new_height > canvas_height:
                    new_height = canvas_height - top_left[1] - 10

                scaled_layer = dict(layer)
                scaled_layer['bottom_right'] = [top_left[0] + new_width, top_left[1] + new_height]
                updated_layers.append(scaled_layer)
            else:
                updated_layers.append(layer)
        else:
            updated_layers.append(layer)

    return updated_layers


def validate_and_fix_layout_bounds(
    layers: List[Dict],
    canvas_width: int = 896,
    canvas_height: int = 2240
) -> List[Dict]:
    """
    Validate and fix layout bounds to ensure all elements are within canvas.
    """
    updated_layers = []

    for layer in layers:
        top_left = layer['top_left']
        bottom_right = layer['bottom_right']

        fixed_top_left = [
            max(0, min(top_left[0], canvas_width - 50)),
            max(0, min(top_left[1], canvas_height - 50))
        ]

        fixed_bottom_right = [
            max(fixed_top_left[0] + 50, min(bottom_right[0], canvas_width)),
            max(fixed_top_left[1] + 50, min(bottom_right[1], canvas_height))
        ]

        fixed_layer = dict(layer)
        fixed_layer['top_left'] = fixed_top_left
        fixed_layer['bottom_right'] = fixed_bottom_right
        updated_layers.append(fixed_layer)

    return updated_layers


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_bbox_area(bbox: Dict) -> int:
    """Calculate area of a bounding box."""
    top_left = bbox['top_left']
    bottom_right = bbox['bottom_right']
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]

    if width <= 0 or height <= 0:
        return 0

    return width * height


def bboxes_overlap(bbox1: Dict, bbox2: Dict, threshold: float = 0.3) -> bool:
    """Check if two bounding boxes overlap significantly."""
    x1_min, y1_min = bbox1['top_left']
    x1_max, y1_max = bbox1['bottom_right']
    x2_min, y2_min = bbox2['top_left']
    x2_max, y2_max = bbox2['bottom_right']

    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap

    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)

    min_area = min(area1, area2)
    return (intersection / min_area) > threshold if min_area > 0 else False


def select_largest_non_overlapping_bboxes(
    bboxes: List[Dict],
    category: str,
    count: int
) -> List[Dict]:
    """
    Select the largest non-overlapping bounding boxes of a specific category.
    """
    filtered = [b for b in bboxes if b.get('category') == category]
    filtered.sort(key=calculate_bbox_area, reverse=True)

    selected = []
    for bbox in filtered:
        if not any(bboxes_overlap(bbox, s) for s in selected):
            selected.append(bbox)
            if len(selected) >= count:
                break

    return selected


def get_random_colors(color_idx: Dict, num_colors: int) -> List[str]:
    """Get random colors from color index."""
    colors = list(color_idx.keys())
    colors = [c for c in colors if c not in ['white']]
    return random.sample(colors, min(num_colors, len(colors)))


def extract_images_from_caption(full_caption: str) -> List[str]:
    """
    Extract image descriptions from caption using new format.
    Format: Figure: description.
    """
    image_elements = []

    pattern = r'Figure:\s+([^.]+\.)'
    matches = re.findall(pattern, full_caption, re.IGNORECASE)

    for description in matches:
        image_elements.append({
            'description': description.strip()
        })

    return image_elements


def extract_images_from_figures(figures: List[Dict]) -> List[Dict]:
    """
    Extract image descriptions from figures data by randomly selecting from ideas.
    """
    image_elements = []

    for figure in figures:
        ideas = figure.get('ideas', [])
        if ideas:
            selected_idea = random.choice(ideas)
            image_elements.append({
                'description': selected_idea.strip()
            })
        else:
            image_elements.append({
                'description': "A simple abstract illustration relevant to the content."
            })

    return image_elements


def extract_text_elements(full_caption: str) -> List[str]:
    """
    Extract text content from captions (quoted text).
    """
    text_elements = []

    pattern = r'"([^"]+)"'
    matches = re.findall(pattern, full_caption)

    for text_content in matches:
        text_content = text_content.strip()
        if text_content not in text_elements:
            text_elements.append(text_content)

    return text_elements


def clean_caption_text(caption: str) -> str:
    """
    Clean up caption text - remove "Figure: " prefix and normalize spacing.
    """
    caption = re.sub(r'Figure:\s+[^.]+\.', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\s+', ' ', caption).strip()
    return caption


def extract_font_color_from_bboxes(bboxes: List[Dict], font_idx: Dict) -> Tuple[str, List[int]]:
    """
    Extract font and colors from text bboxes in the layout.
    Returns a tuple of (font_token, list_of_color_ids).
    If no English font is found, returns a random English font.
    """
    text_bboxes = [b for b in bboxes if b.get('category') == 'text']

    fonts_found = []
    colors_found = []

    for bbox in text_bboxes:
        font_color_info = bbox.get('font_color_info', '')
        if font_color_info:
            font_match = re.search(r'<(en-font-\d+)>', font_color_info)
            if font_match:
                fonts_found.append(font_match.group(1))

            color_match = re.search(r'<color-(\d+)>', font_color_info)
            if color_match:
                colors_found.append(int(color_match.group(1)))

    font_token = None
    if fonts_found:
        font_token = max(set(fonts_found), key=fonts_found.count)
    else:
        en_fonts = [k for k in font_idx.keys() if k.startswith('en-')]
        if en_fonts:
            selected_font = random.choice(en_fonts)
            font_id = font_idx[selected_font]
            font_token = f'en-font-{font_id}'
        else:
            font_token = 'en-font-0'

    unique_color_ids = list(set(colors_found))

    return font_token, unique_color_ids


def get_color_name_from_id(color_id: int, color_idx: Dict) -> str:
    """Get color name from color ID."""
    for name, cid in color_idx.items():
        if cid == color_id:
            return name
    return 'black'


def sort_by_reading_order(bboxes: List[Dict]) -> List[Dict]:
    """
    Sort bboxes by reading order (left to right, top to bottom).
    """
    def reading_order_key(bbox):
        x, y = bbox['top_left']
        return (y, x)

    return sorted(bboxes, key=reading_order_key)


def text_overlaps_with_image(text_bbox: Dict, image_bbox: Dict, threshold: float = 0.1) -> bool:
    """
    Check if a text bbox overlaps with an image bbox.
    """
    return bboxes_overlap(text_bbox, image_bbox, threshold)


def count_text_overlaps_per_image(text_bboxes: List[Dict], image_bboxes: List[Dict]) -> Dict[int, int]:
    """
    Count how many text bboxes overlap with each image bbox.
    """
    overlap_counts = {}

    for img_idx, img_bbox in enumerate(image_bboxes):
        overlap_count = 0
        for txt_bbox in text_bboxes:
            if text_overlaps_with_image(txt_bbox, img_bbox):
                overlap_count += 1
        overlap_counts[img_idx] = overlap_count

    return overlap_counts


def select_non_overlapping_text_bboxes(
    text_bboxes: List[Dict],
    image_bboxes: List[Dict],
    required_count: int
) -> List[Dict]:
    """
    Select text bboxes that don't overlap with any image bboxes.
    """
    selected_text = []

    for txt_bbox in text_bboxes:
        overlaps_with_image = any(
            text_overlaps_with_image(txt_bbox, img_bbox)
            for img_bbox in image_bboxes
        )

        if not overlaps_with_image:
            overlaps_with_text = any(
                bboxes_overlap(txt_bbox, selected_txt)
                for selected_txt in selected_text
            )

            if not overlaps_with_text:
                selected_text.append(txt_bbox)
                if len(selected_text) >= required_count:
                    break

    return selected_text


def validate_input_format(data: List[Dict]) -> Tuple[bool, str]:
    """
    Validate the input data format.
    """
    if not isinstance(data, list):
        return False, "Input data should be a list"

    if not data:
        return False, "Input data is empty"

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            return False, f"Entry {i} is not a dictionary"

        required_fields = ['full_image_caption', 'background_caption', 'figures']
        for field in required_fields:
            if field not in entry:
                return False, f"Entry {i} missing required field: {field}"

        if not entry['full_image_caption'].strip():
            return False, f"Entry {i} has empty full_image_caption"

        if not isinstance(entry['figures'], list):
            return False, f"Entry {i} figures field should be a list"

    return True, "Format is valid"


# ============================================================================
# STAGE PROCESSING FUNCTIONS (MATCHING)
# ============================================================================

def match_text_to_bboxes(
    text_elements: List[str],
    candidate_text_bboxes: List[Dict],
    image_bboxes: List[Dict] = None
) -> List[Dict]:
    """
    Match text elements to optimal bounding boxes using improved algorithm.
    """
    if not text_elements or not candidate_text_bboxes:
        return []

    image_bboxes = image_bboxes or []

    matched_pairs = find_best_text_positions(
        text_elements,
        candidate_text_bboxes,
        image_bboxes,
        margin=20
    )

    matched_elements = []
    for i, (bbox, text_content) in enumerate(matched_pairs):
        matched_elements.append({
            'id': i + 1,
            'summary': text_content,
            'bbox': bbox
        })

    return matched_elements


def match_figures_to_bboxes(
    figure_elements: List[Dict],
    candidate_element_bboxes: List[Dict],
    reserved_text_areas: List[Dict] = None
) -> List[Dict]:
    """
    Match figure elements to optimal bounding boxes using smart selection.
    """
    if not figure_elements or not candidate_element_bboxes:
        return []

    matched_pairs = smart_image_selection(
        figure_elements,
        candidate_element_bboxes,
        reserved_text_areas or []
    )

    matched_elements = []
    for i, (bbox, description) in enumerate(matched_pairs):
        matched_elements.append({
            'id': i + 1,
            'description': description,
            'bbox': bbox
        })

    return matched_elements


def stage_3_compose_with_bbox(
    llm: LLM,
    text_elements: List[Dict],
    figure_elements: List[Dict],
    stage_c_tmpl_text: str,
    canvas_width: int = 896,
    canvas_height: int = 2240,
    sents_enum: List[Dict] = None
) -> str:
    """
    Compose final infographic description using text and figure elements with bbox information.
    """
    prompt = render_template(
        stage_c_tmpl_text,
        text_elements=text_elements,
        figure_elements=figure_elements,
        canvas_width=canvas_width,
        canvas_height=canvas_height
    )

    response = llm.generate_single(
        prompt,
        enable_thinking=False
    )

    final_str = " ".join(response.strip().split())

    if len(final_str) < 50 or not final_str:
        print("Warning: Stage 3 output too short, creating fallback caption")
        if sents_enum:
            sentence_texts = [sent["text"] for sent in sents_enum[:3]]
            final_str = f"The image is an infographic that presents information about the following content: {' '.join(sentence_texts)}"
        else:
            title = "Information Overview"
            fallback_parts = [f'The image is an infographic titled "{title}" with a clean white background.']

            for text_elem in text_elements:
                fallback_parts.append(f'The text "{text_elem.get("summary", "")}" is positioned in the layout.')

            for fig_elem in figure_elements:
                fallback_parts.append(f'A visual element showing {fig_elem.get("description", "")} is included.')

            fallback_parts.append("The overall style of the image is clean and informative.")
            final_str = " ".join(fallback_parts)

    return final_str


# ============================================================================
# LAYOUT CREATION HELPERS
# ============================================================================

def create_layout_elements_from_stage1(
    stage_1_result: Dict[str, Any],
    bboxes: List[Dict],
    title: str = None
) -> List[Dict]:
    """
    Create layout elements by matching Stage 1 content (title + segments + figures) to bboxes
    Using optimized bbox assignment to avoid overlaps
    """
    layout_elements = []

    stage_1_title = stage_1_result.get('title', '')
    final_title = stage_1_title if stage_1_title else title

    text_bboxes = [b for b in bboxes if b.get('category') == 'text' and b.get('bottom_right') != [896, 2240]]
    element_bboxes = [b for b in bboxes if b.get('category') == 'element' and b.get('bottom_right') != [896, 2240]]

    text_contents = []
    if final_title:
        text_contents.append(final_title)

    segments = stage_1_result.get('segments', [])
    for segment in segments:
        text_contents.append(segment['text'])

    optimized_text_pairs = find_best_text_positions(
        text_contents,
        text_bboxes,
        element_bboxes,
        margin=25
    )

    for i, (bbox, text_content) in enumerate(optimized_text_pairs):
        is_title = (i == 0 and final_title)
        layout_elements.append({
            'category': 'text',
            'content': text_content,
            'bbox': bbox,
            'is_title': is_title
        })

    reserved_text_areas = [elem['bbox'] for elem in layout_elements if elem['category'] == 'text']

    figures = stage_1_result.get('figures', [])
    figure_elements = []
    for figure in figures:
        figure_elements.append({
            'description': figure.get('description', 'decorative element')
        })

    optimized_figure_pairs = smart_image_selection(
        figure_elements,
        element_bboxes,
        reserved_text_areas
    )

    for bbox, description in optimized_figure_pairs:
        layout_elements.append({
            'category': 'element',
            'content': description,
            'bbox': bbox
        })

    return layout_elements

def create_wiki_layout_from_layout_elements(
    layout_elements: List[Dict],
    full_image_caption: str,
    bbox_set: Dict,
    wiki_id: int
) -> Dict:
    """
    Create wiki layout structure from layout elements following the correct format
    """
    cleaned_caption = clean_caption_text(full_image_caption)

    output_layers = []

    base_layer = {
        'category': 'base',
        'top_left': [0, 0],
        'bottom_right': [896, 2240],
        'caption': full_image_caption
    }
    output_layers.append(base_layer)

    bboxes = bbox_set.get('bboxes', [])
    background_bboxes = [b for b in bboxes if b.get('category') == 'background']
    if background_bboxes:
        background_layer = background_bboxes[0].copy()
        output_layers.append(background_layer)
    else:
        fallback_bg_layer = {
            'category': 'background',
            'top_left': [0, 0],
            'bottom_right': [896, 2240],
            'caption': 'Clean white background with professional layout'
        }
        output_layers.append(fallback_bg_layer)

    text_elements = [elem for elem in layout_elements if elem['category'] == 'text']
    element_elements = [elem for elem in layout_elements if elem['category'] == 'element']

    for elem in element_elements:
        bbox = elem['bbox']
        element_layer = {
            'category': 'element',
            'top_left': bbox['top_left'],
            'bottom_right': bbox['bottom_right'],
            'caption': elem.get('content', 'decorative element')
        }
        output_layers.append(element_layer)

    default_fonts = ['en-font-101', 'en-font-15', 'en-font-154']
    default_color = 'color-1'

    for i, elem in enumerate(text_elements):
        bbox = elem['bbox']
        text_content = elem.get('content', '')
        is_title = elem.get('is_title', False)

        if is_title:
            selected_font = default_fonts[0]
        else:
            selected_font = default_fonts[(i % len(default_fonts))]

        caption = f'Text "{text_content}" in <{default_color}>, <{selected_font}>. '

        text_layer = {
            'category': 'text',
            'top_left': bbox['top_left'],
            'bottom_right': bbox['bottom_right'],
            'caption': caption,
            'text': text_content
        }

        output_layers.append(text_layer)

    output_layers = auto_scale_small_text_bboxes(output_layers, canvas_width=896, canvas_height=2240)
    output_layers = validate_and_fix_layout_bounds(output_layers, canvas_width=896, canvas_height=2240)
    layout_quality = validate_layout_quality(output_layers)

    wiki_result = {
        'index': wiki_id,
        'layers_all': output_layers,
        'full_image_caption': full_image_caption,
        'original_bbox_index': bbox_set.get('index', -1),
        'layout_quality': layout_quality
    }

    return wiki_result

# ============================================================================
# WIKI LAYOUT CREATION
# ============================================================================

def create_wiki_layout_from_elements(
    matched_text_elements: List[Dict],
    matched_figure_elements: List[Dict],
    full_image_caption: str,
    bbox_set: Dict,
    wiki_id: int
) -> Dict:
    """
    Create wiki layout format from matched elements, following merge_stage_narrator.py format.
    """
    bboxes = bbox_set.get('bboxes', [])

    cleaned_caption = clean_caption_text(full_image_caption)

    output_layers = []

    base_layer = {
        'category': 'base',
        'top_left': [0, 0],
        'bottom_right': [896, 2240],
        'caption': cleaned_caption
    }
    output_layers.append(base_layer)

    background_bboxes = [b for b in bboxes if b.get('category') == 'background']
    if len(background_bboxes) > 0:
        bg_bbox = background_bboxes[0]
        bg_layer = {
            'category': 'background',
            'top_left': bg_bbox['top_left'],
            'bottom_right': bg_bbox['bottom_right'],
            'caption': 'Clean white background with professional layout'
        }
    else:
        bg_layer = {
            'category': 'background',
            'top_left': [0, 0],
            'bottom_right': [896, 2240],
            'caption': 'Clean white background with professional layout'
        }
    output_layers.append(bg_layer)

    for fig_elem in matched_figure_elements:
        if 'bbox' in fig_elem:
            bbox = fig_elem['bbox']
            element_layer = {
                'category': 'element',
                'top_left': bbox['top_left'],
                'bottom_right': bbox['bottom_right'],
                'caption': fig_elem.get('description', 'decorative element')
            }
            output_layers.append(element_layer)

    default_fonts = ['en-font-101', 'en-font-15', 'en-font-154']
    default_color = 'color-1'

    for i, text_elem in enumerate(matched_text_elements):
        if 'bbox' in text_elem:
            bbox = text_elem['bbox']
            text_content = text_elem.get('summary', '')

            selected_font = default_fonts[i % len(default_fonts)]

            caption = f'Text "{text_content}" in <{default_color}>, <{selected_font}>. '

            text_layer = {
                'category': 'text',
                'top_left': bbox['top_left'],
                'bottom_right': bbox['bottom_right'],
                'caption': caption,
                'text': text_content
            }

            output_layers.append(text_layer)

    output_layers = auto_scale_small_text_bboxes(output_layers, canvas_width=896, canvas_height=2240)
    output_layers = validate_and_fix_layout_bounds(output_layers, canvas_width=896, canvas_height=2240)
    layout_quality = validate_layout_quality(output_layers)

    wiki_result = {
        'index': wiki_id,
        'layers_all': output_layers,
        'full_image_caption': full_image_caption,
        'original_bbox_index': bbox_set.get('index', -1),
        'layout_quality': layout_quality
    }

    return wiki_result


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_sample_with_bbox_matching(
    llm: LLM,
    item: Dict[str, Any],
    stage_1_path: str,
    stage_2_path: str,
    extracted_bboxes: List[Dict],
    infographic_id: int,
    max_retries: int = 2
) -> Optional[Dict]:
    """
    Process a single sample through 2-stage pipeline with bbox matching and keyword checking.
    """
    ensure_file(stage_1_path, "Stage 1")
    ensure_file(stage_2_path, "Stage 2")

    stage_1_tmpl_text = read_text(stage_1_path)
    stage_2_tmpl_text = read_text(stage_2_path)

    qa_pairs = item.get('qa_pairs', [])
    keywords = extract_answer_keywords(qa_pairs)

    has_answers = has_answerable_questions(qa_pairs)

    if not has_answers:
        print(f"Processing infographic {infographic_id}, No answerable questions (Squad v2 impossible questions) - skipping keyword check")
    else:
        print(f"Processing infographic {infographic_id}, Keywords to check: {keywords}")

    for retry_count in range(max_retries + 1):
        try:
            if retry_count > 0:
                print(f"  Retry attempt {retry_count}/{max_retries}")

            context = item.get('context', '')
            if not context:
                raise ValueError("No context provided")

            stage_1_result = stage_1_generate_content(llm, context, qa_pairs, stage_1_tmpl_text)

            if not extracted_bboxes:
                raise ValueError("No bboxes available")

            bbox_set = random.choice(extracted_bboxes)
            bboxes = bbox_set.get('bboxes', [])

            layout_elements = create_layout_elements_from_stage1(
                stage_1_result,
                bboxes
            )

            full_image_caption = stage_2_generate_caption(
                llm,
                layout_elements,
                stage_2_tmpl_text,
                canvas_width=896,
                canvas_height=2240
            )

            if has_answers:
                keywords_found, found_keywords = check_keywords_in_caption(full_image_caption, keywords)
                if not keywords_found:
                    if retry_count < max_retries:
                        print(f"    Keywords not found in caption, retrying... (attempt {retry_count + 1})")
                        continue
                    else:
                        print(f"    Keywords not found after {max_retries + 1} attempts, skipping sample")

                        debug_data = {
                            'infographic_id': infographic_id,
                            'keywords_to_find': keywords,
                            'generated_caption': full_image_caption,
                            'layout_elements': layout_elements,
                            'context': item.get('context', ''),
                            'qa_pairs': qa_pairs,
                            'stage_1_result': stage_1_result,
                            'caption_lower': full_image_caption.lower(),
                            'keywords_lower': [k.lower() for k in keywords]
                        }

                        debug_file = f"debug_failed_sample_{infographic_id}.json"
                        save_json(debug_file, debug_data)
                        print(f"    DEBUG: Saved failed sample to {debug_file}")

                        return None
                else:
                    print(f"    Keywords found: {found_keywords[:3]}...")

            wiki_layout = create_wiki_layout_from_layout_elements(
                layout_elements,
                full_image_caption,
                bbox_set,
                infographic_id
            )

            return {
                'infographic_id': infographic_id,
                'wiki_layout': wiki_layout,
                'full_image_caption': full_image_caption,
                'background_caption': 'Clean white background with professional layout',
                'stage_1_result': stage_1_result,
                'layout_elements': layout_elements,
                'bbox_set_index': bbox_set.get('index', -1),
                'success': True,
                'skipped_keyword_check': not has_answers
            }

        except Exception as e:
            print(f"    Error in attempt {retry_count + 1}: {str(e)}")
            if retry_count < max_retries:
                print(f"    Retrying...")
                continue
            else:
                print(f"    Failed after {max_retries + 1} attempts")
                return {
                    'infographic_id': infographic_id,
                    'success': False,
                    'error': str(e),
                    'skipped_keyword_check': False
                }

    return None


# ============================================================================
# LEGACY FUNCTIONS (kept for compatibility)
# ============================================================================

def merge_narrator_data(
    infographic_generated: List[Dict],
    extracted_bboxes: List[Dict],
    color_idx: Dict,
    font_idx: Dict,
    start_wiki_idx: int = 0
) -> List[Dict]:
    """
    Process narrator-generated infographic data and merge with bboxes.
    Same as original but handles simplified input format.
    """
    result = []

    bbox_by_index = {item['index']: item for item in extracted_bboxes}
    available_indices = list(bbox_by_index.keys())

    for wiki_idx, infographic in enumerate(infographic_generated):
        wiki_id = start_wiki_idx + wiki_idx + 1

        generated_infographic = infographic.get('generated_infographic', {})

        if generated_infographic is None:
            print(f"Warning: generated_infographic is None for wiki {wiki_id}, skipping")
            continue

        if isinstance(generated_infographic, str):
            try:
                generated_infographic = json.loads(generated_infographic)
            except:
                print(f"Warning: Could not parse generated_infographic for wiki {wiki_id}")
                continue

        if not isinstance(generated_infographic, dict):
            print(f"Warning: generated_infographic is not a dict for wiki {wiki_id}, skipping")
            continue

        full_image_caption = generated_infographic.get('full_image_caption', '')
        background_caption = generated_infographic.get('background_caption', '')
        figures = generated_infographic.get('figures', [])

        if not full_image_caption:
            print(f"Warning: No full_image_caption for wiki {wiki_id}, skipping")
            continue

        image_elements = extract_images_from_figures(figures)
        text_elements = extract_text_elements(full_image_caption)

        if not available_indices:
            print("Warning: No more bbox indices available, wrapping around")
            available_indices = list(bbox_by_index.keys())
            print(f"  Reset available indices count: {len(available_indices)}")

        selected_bbox_index = random.choice(available_indices)
        available_indices.remove(selected_bbox_index)

        bbox_data = bbox_by_index[selected_bbox_index]
        bboxes = bbox_data['bboxes']

        font_token, layout_color_ids = extract_font_color_from_bboxes(bboxes, font_idx)

        layout_colors = []
        for color_id in layout_color_ids:
            color_name = get_color_name_from_id(color_id, color_idx)
            if color_name != 'white':
                layout_colors.append(color_name)

        if not layout_colors:
            num_colors = random.randint(1, 4)
            layout_colors = get_random_colors(color_idx, num_colors)

        background_bboxes = [b for b in bboxes if b.get('category') == 'background']

        element_bboxes = [b for b in bboxes if b.get('category') == 'element']
        full_image_elements = [b for b in element_bboxes if b['bottom_right'] == [896, 2240]]
        regular_elements = [b for b in element_bboxes if b['bottom_right'] != [896, 2240]]

        cleaned_caption = clean_caption_text(full_image_caption)

        output_layers = []

        base_layer = {
            'category': 'base',
            'top_left': [0, 0],
            'bottom_right': [896, 2240],
            'caption': cleaned_caption
        }
        output_layers.append(base_layer)

        if len(background_bboxes) > 0:
            bg_bbox = background_bboxes[0]
            bg_layer = {
                'category': 'element',
                'top_left': bg_bbox['top_left'],
                'bottom_right': bg_bbox['bottom_right'],
                'caption': bg_bbox.get('caption', '')
            }
            output_layers.append(bg_layer)
        else:
            bg_layer = {
                'category': 'element',
                'top_left': [0, 0],
                'bottom_right': [896, 2240],
                'caption': "The image you've provided is completely blank and white. There are no objects, no text, no colors, and no discernible features. It's a simple, unadorned white background with no additional elements."
            }
            output_layers.append(bg_layer)

        num_available_elements = len(regular_elements)
        num_figures_to_use = min(len(image_elements), num_available_elements)

        image_elements_to_use = image_elements[:num_figures_to_use]

        regular_elements.sort(key=calculate_bbox_area, reverse=True)
        selected_decorative = []

        for bbox in regular_elements:
            overlaps = any(bboxes_overlap(bbox, s) for s in selected_decorative)
            if not overlaps:
                selected_decorative.append(bbox)
                if len(selected_decorative) >= num_figures_to_use:
                    break

        bbox_caption_pairs = []
        for idx, bbox in enumerate(selected_decorative):
            if idx < len(image_elements_to_use):
                caption = image_elements_to_use[idx]['description']
            else:
                caption = "decorative element"
            bbox_caption_pairs.append((bbox, caption))

        bbox_caption_pairs.sort(key=lambda pair: (pair[0]['top_left'][1], pair[0]['top_left'][0]))

        for bbox, caption in bbox_caption_pairs:
            output_layer = {
                'category': 'element',
                'top_left': bbox['top_left'],
                'bottom_right': bbox['bottom_right'],
                'caption': caption
            }
            output_layers.append(output_layer)

        valid_text_bboxes = [b for b in bboxes if b.get('category') == 'text' and b.get('bottom_right') != [896, 2240]]
        valid_text_bboxes.sort(key=calculate_bbox_area, reverse=True)

        optimized_text_pairs = find_best_text_positions(
            text_elements,
            valid_text_bboxes,
            selected_decorative,
            margin=25
        )

        if len(optimized_text_pairs) < len(text_elements):
            remaining_text = text_elements[len(optimized_text_pairs):]

            reserved_text_areas = [pair[0] for pair in optimized_text_pairs]
            optimized_image_pairs = smart_image_selection(
                image_elements_to_use,
                selected_decorative,
                reserved_text_areas
            )

            selected_decorative = [pair[0] for pair in optimized_image_pairs]

            additional_text_pairs = find_best_text_positions(
                remaining_text,
                valid_text_bboxes[len(optimized_text_pairs):],
                selected_decorative,
                margin=25
            )

            optimized_text_pairs.extend(additional_text_pairs)

        for text_bbox, text_content in optimized_text_pairs:

            bbox_font_color_info = text_bbox.get('font_color_info', '')

            bbox_font_token = font_token
            if bbox_font_color_info:
                font_match = re.search(r'<(en-font-\d+)>', bbox_font_color_info)
                if font_match:
                    bbox_font_token = font_match.group(1)
                else:
                    en_fonts = [k for k in font_idx.keys() if k.startswith('en-')]
                    if en_fonts:
                        selected_font = random.choice(en_fonts)
                        font_id = font_idx[selected_font]
                        bbox_font_token = f'en-font-{font_id}'

            color_name = 'black'
            if bbox_font_color_info:
                color_match = re.search(r'<color-(\d+)>', bbox_font_color_info)
                if color_match:
                    color_id = color_match.group(1)
                    color_name = get_color_name_from_id(int(color_id), color_idx)
                    if color_name == 'white':
                        color_name = random.choice(layout_colors) if layout_colors else 'black'
                else:
                    color_name = random.choice(layout_colors) if layout_colors else 'black'
            else:
                color_name = random.choice(layout_colors) if layout_colors else 'black'

            color_id = color_idx[color_name]
            caption = f'Text "{text_content}" in <color-{color_id}>, <{bbox_font_token}>. '
            output_layer = {
                'category': 'text',
                'top_left': text_bbox['top_left'],
                'bottom_right': text_bbox['bottom_right'],
                'caption': caption,
                'text': text_content
            }
            output_layers.append(output_layer)

        print(f"Auto-scaling small text bboxes for wiki {wiki_id}...")
        output_layers = auto_scale_small_text_bboxes(output_layers, canvas_width=896, canvas_height=2240)

        print(f"Validating canvas bounds for wiki {wiki_id}...")
        output_layers = validate_and_fix_layout_bounds(output_layers, canvas_width=896, canvas_height=2240)

        layout_quality = validate_layout_quality(output_layers)

        if not layout_quality['passes_quality']:
            print(f"Layout quality warning for wiki {wiki_id}: score={layout_quality['quality_score']:.2f}")
        else:
            print(f"Good layout quality for wiki {wiki_id}: score={layout_quality['quality_score']:.2f}")

        result_item = {
            'index': wiki_id,
            'layers_all': output_layers,
            'full_image_caption': full_image_caption,
            'original_bbox_index': selected_bbox_index,
            'layout_quality': layout_quality
        }
        result.append(result_item)

    return result


def main():
    """
    Main function for processing narrator data with bbox matching pipeline.
    """
    parser = argparse.ArgumentParser(description='Generate 2-stage infographic data with bbox matching using Qwen3 or ChatGPT')

    # Backend selection
    parser.add_argument('--backend', type=str, choices=['qwen3', 'gpt'], default='qwen3',
                        help='LLM backend to use: qwen3 (local) or gpt (ChatGPT API)')

    # Qwen3 args
    parser.add_argument('--model_name', type=str, default='unsloth/Qwen3-8B',
                        help='Model name or path (for Qwen3 backend)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization (Qwen3)')

    # GPT args
    parser.add_argument('--gpt_model', type=str, default='gpt-4o',
                        help='ChatGPT model name (when --backend gpt)')
    parser.add_argument('--system_prompt', type=str, default=None,
                        help='Optional system prompt for ChatGPT backend')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')

    # Common generation args
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling parameter (used by Qwen3)')
    parser.add_argument('--max_tokens', type=int, default=8192,
                        help='Maximum tokens to generate (Qwen3 uses *2 inside; GPT uses this value)')

    # Data / templates / IO
    parser.add_argument('--input_data', type=str,
                        default='/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl',
                        help='Path to Squad v2 JSONL file')
    parser.add_argument('--stage_1', type=str,
                        default='./src/prompts/content_des_stage_1.jinja',
                        help='Path to Stage 1 Jinja template')
    parser.add_argument('--stage_2', type=str,
                        default='./src/prompts/content_des_stage_2.jinja',
                        help='Path to Stage 2 Jinja template')
    parser.add_argument('--extracted_bboxes', type=str,
                        default='./src/data/narrator/extracted_bboxes.json',
                        help='Path to extracted bboxes JSON file')
    parser.add_argument('--output_dir', type=str,
                        default='./src/data/create_data/output/narrator_format',
                        help='Output directory for wiki files')

    # Control
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')
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
    print("Initializing 2-Stage Wiki Layout Generation with BBox Matching")
    print("="*60)

    # Load input data
    print(f"\n[1/5] Loading input data from: {args.input_data}")
    input_data_full = load_squad_v2_data(args.input_data)
    print(f"Total unique contexts loaded: {len(input_data_full)}")

    if input_data_full:
        sample_item = input_data_full[0]
        sample_keywords = extract_answer_keywords(sample_item.get('qa_pairs', []))
        print(f"Sample keywords from first context: {sample_keywords[:5]}")

    chunk_size = 50
    start_data_idx = (args.start - 1) * chunk_size
    end_data_idx = (args.end - 1) * chunk_size if args.end is not None else len(input_data_full)

    max_files_needed = (len(input_data_full) + chunk_size - 1) // chunk_size
    if args.start < 1:
        raise ValueError("Start file index must be >= 1")
    if args.end is not None and args.end <= args.start:
        raise ValueError("End file index must be > start file index")
    if args.start > max_files_needed:
        raise ValueError(f"Start file index {args.start} exceeds available data (max files: {max_files_needed})")

    print(f"File indices: {args.start} to {args.end if args.end else 'end'}")
    print(f"Data indices: {start_data_idx} to {end_data_idx}")
    print(f"Unique contexts per file: {chunk_size}")

    input_data_sliced = input_data_full[start_data_idx:end_data_idx]
    print(f"Sliced data from data index {start_data_idx} to {end_data_idx}: {len(input_data_sliced)} samples")

    if args.num_samples:
        input_data = input_data_sliced[:args.num_samples]
        print(f"Further limited to {args.num_samples} samples")
    else:
        input_data = input_data_sliced
        print(f"Processing all {len(input_data)} samples from slice")

    # Load extracted bboxes
    print(f"\n[2/5] Loading extracted bboxes from: {args.extracted_bboxes}")
    try:
        with open(args.extracted_bboxes, 'r', encoding='utf-8') as f:
            extracted_bboxes = json.load(f)
        print(f"  - Loaded {len(extracted_bboxes)} bbox sets")
        print(f"  - Using fixed fonts: en-font-101, en-font-15, en-font-154")
        print(f"  - Using fixed color: color-1 (black) for all text")
    except Exception as e:
        print(f"Error loading extracted bboxes: {e}")
        return

    # Initialize LLM backend
    print(f"\n[3/5] Initializing LLM backend: {args.backend}")

    if args.backend == 'qwen3':
        print(f"Using Qwen3 model: {args.model_name}")
        llm: LLM = Qwen3Inference(
            model_name=args.model_name,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="auto",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens * 2
        )
    else:
        print(f"Using ChatGPT model: {args.gpt_model}")
        llm = ChatGPTInference(
            model_name=args.gpt_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_prompt=args.system_prompt,
            api_key=args.openai_api_key
        )

    # Create output directory
    print(f"\n[4/5] Setting up output directory: {args.output_dir}")
    ensure_dir(args.output_dir)

    # Process data
    print(f"\n[5/5] Processing samples through 2-stage pipeline with bbox matching")
    print(f"Templates:")
    print(f"  Stage 1: {args.stage_1}")
    print(f"  Stage 2: {args.stage_2}")
    print(f"Max retries: {args.max_retries}")
    print("-"*60)

    results = []
    saved_files = []
    total_processed = 0
    successful_count = 0
    failed_count = 0
    keyword_failed_count = 0
    skipped_keyword_check_count = 0

    for i, item in enumerate(tqdm(input_data, desc="Processing samples")):
        infographic_id = start_data_idx + i + 1

        result = process_sample_with_bbox_matching(
            llm,
            item,
            args.stage_1,
            args.stage_2,
            extracted_bboxes,
            infographic_id,
            args.max_retries
        )

        if result is None:
            keyword_failed_count += 1
            failed_count += 1
        elif result["success"]:
            successful_count += 1
            if result.get("skipped_keyword_check", False):
                skipped_keyword_check_count += 1
        else:
            failed_count += 1

        results.append(result)
        total_processed += 1

        if len(results) >= chunk_size:
            valid_results = [r for r in results if r is not None]
            if valid_results:
                first_infographic_id = valid_results[0]['infographic_id']
                file_index = (first_infographic_id - 1) // chunk_size + 1
            else:
                file_index = (start_data_idx + i - len(results) + 2) // chunk_size + 1

            filename = save_chunk_to_file(results, args.output_dir, file_index)
            if filename:
                saved_files.append(filename)
            results = []

    if results:
        print("\n" + "="*60)
        print("Saving final chunk")
        print("="*60)
        valid_results = [r for r in results if r is not None]
        if valid_results:
            first_infographic_id = valid_results[0]['infographic_id']
            file_index = (first_infographic_id - 1) // chunk_size + 1
        else:
            file_index = (start_data_idx + total_processed - len(results) + 1) // chunk_size + 1

        filename = save_chunk_to_file(results, args.output_dir, file_index)
        if filename:
            saved_files.append(filename)

    print("\n" + "="*60)
    print("2-Stage Wiki Layout Generation with BBox Matching Complete - Final Statistics")
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
    print(f"Failed (keywords not found): {keyword_failed_count} ({(keyword_failed_count)/total_processed*100:.1f}%)")
    print(f"Total failed: {failed_count} ({failed_count/total_processed*100:.1f}%)")
    print(f"Output directory: {args.output_dir}")

    if failed_count > 0:
        print(f"\nNote: Failed samples were still saved to maintain indexing consistency.")
        print(f"Check individual file contents to see which samples failed.")

    if skipped_keyword_check_count > 0:
        print(f"\nNote: {skipped_keyword_check_count} samples had no answerable questions (Squad v2 impossible questions).")
        print(f"These were processed successfully but keyword checking was skipped.")

    print(f"\nFiles saved:")
    for filename in saved_files:
        print(f"  - {filename}")

    print("\n" + "="*60)
    print("2-Stage Wiki Layout Generation with BBox Matching complete!")
    print("="*60)


if __name__ == '__main__':
    main()
