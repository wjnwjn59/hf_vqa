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
        print(prompt)

        resp = self.client.responses.create(
            model=self.model_name,
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

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def render_template(tmpl_text: str, **kwargs) -> str:
    return Template(tmpl_text, trim_blocks=True, lstrip_blocks=True).render(**kwargs).strip()


# ============================================================================
# COLOR AND BBOX UTILITIES
# ============================================================================

def load_color_mapping(color_file_path: str) -> Dict[str, int]:
    """Load color name to index mapping from JSON file"""
    with open(color_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_color_name_from_idx(color_idx: int, color_mapping: Dict[str, int]) -> str:
    """Get color name from color index"""
    for color_name, idx in color_mapping.items():
        if idx == color_idx:
            return color_name
    return "black"  # fallback

def calculate_bbox_area(bbox_dict: Dict) -> int:
    """Calculate area of a bounding box from dict with top_left and bottom_right"""
    top_left = bbox_dict.get('top_left', [0, 0])
    bottom_right = bbox_dict.get('bottom_right', [0, 0])
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    return max(0, width * height)

def filter_text_bboxes_by_area(bboxes: List[Dict], min_area: int = 10000) -> List[Dict]:
    """Filter text bboxes to keep only those with area >= min_area pixels"""
    filtered = []
    for bbox in bboxes:
        if bbox.get('category') == 'text':
            area = calculate_bbox_area(bbox)
            if area >= min_area:
                filtered.append(bbox)
        else:
            # Keep non-text bboxes as is
            filtered.append(bbox)
    return filtered


# ============================================================================
# LAYOUT SELECTION AND FORMATTING
# ============================================================================

def select_random_layout(extracted_bboxes: List[Dict], min_text_area: int = 10000) -> Dict:
    """
    Select a random layout from extracted bboxes and filter text areas by minimum size
    """
    if not extracted_bboxes:
        raise ValueError("No bboxes available")
    
    # Select random bbox set
    bbox_set = random.choice(extracted_bboxes)
    bboxes = bbox_set.get('bboxes', [])
    
    # Filter text bboxes by minimum area
    filtered_bboxes = filter_text_bboxes_by_area(bboxes, min_text_area)
    
    return {
        'index': bbox_set.get('index', -1),
        'bboxes': filtered_bboxes
    }

def format_layout_for_prompt(
    bbox_set: Dict, 
    color_mapping: Dict[str, int],
    background_caption: str = "Clean white background with professional layout"
) -> Tuple[str, List[Dict], int, int]:
    """
    Format layout information for model input prompt (new format)
    
    Returns:
        Tuple of (background_caption, layouts_list, n_figures, m_texts)
    """
    bboxes = bbox_set.get('bboxes', [])
    
    layouts = []
    layout_idx = 1
    n_figures = 0
    m_texts = 0
    
    for bbox in bboxes:
        category = bbox.get('category', 'unknown')
        
        if category == 'background' or category == 'base':
            # Skip background and base in layout list as it's handled separately
            continue
        
        # Get bbox coordinates in [x1, y1, x2, y2] format
        top_left = bbox.get('top_left', [0, 0])
        bottom_right = bbox.get('bottom_right', [0, 0])
        bbox_coords = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
        
        layout_item = {
            'idx': layout_idx,
            'type': '',
            'bbox': bbox_coords
        }
        
        if category == 'text':
            layout_item['type'] = 'text'
            m_texts += 1
            
            # Get color information if available
            if 'color' in bbox:
                color_idx = bbox['color']
                color_name = get_color_name_from_idx(color_idx, color_mapping)
                layout_item['color'] = color_name
        elif category == 'element':
            layout_item['type'] = 'figure'
            n_figures += 1
        else:
            # Handle other categories as figure by default
            layout_item['type'] = 'figure'
            n_figures += 1
        
        layouts.append(layout_item)
        layout_idx += 1
    
    return background_caption, layouts, n_figures, m_texts

def format_qa_pairs_for_prompt(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Format QA pairs for model input prompt (new format)
    
    Returns:
        List of dicts with Q and A keys
    """
    if not qa_pairs:
        return []
    
    formatted_qas = []
    
    for qa in qa_pairs:
        question = qa.get('question', '').strip()
        
        # Extract answer text
        answer_text = ""
        answers = qa.get('answers', {})
        
        if isinstance(answers, dict):
            if 'text' in answers:
                if isinstance(answers['text'], list) and answers['text']:
                    answer_text = answers['text'][0]
                else:
                    answer_text = str(answers['text'])
        elif isinstance(answers, list) and answers:
            answer_text = str(answers[0])
        else:
            answer_text = str(answers) if answers else "No answer available"
        
        formatted_qas.append({
            'Q': question,
            'A': answer_text
        })
    
    return formatted_qas


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def load_squad_v2_data(input_path: str) -> List[Dict[str, Any]]:
    """Load Squad v2 data from JSONL file with context deduplication"""
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


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def generate_full_caption_new_format(
    llm: LLM,
    context: str,
    qa_pairs: List[Dict],
    background_caption: str,
    layouts: List[Dict],
    n_figures: int,
    m_texts: int,
    prompt_template_text: str,
    res_w: int,
    res_h: int,
) -> str:
    """
    Generate full image caption using the new format with context, QAs, background, and selected template
    """
    # Format QA pairs for prompt
    qa_list = format_qa_pairs_for_prompt(qa_pairs)
    
    # Render the prompt template
    prompt = render_template(
        prompt_template_text,
        context=context,
        qa_list=qa_list,
        background=background_caption,
        layouts=layouts,
        n_figures=n_figures,
        m_texts=m_texts,
        res_w=res_w,             
        res_h=res_h,            
    )
    
    # Generate response
    response = llm.generate_single(prompt, enable_thinking=False)
    return response.strip()

def extract_text_elements_from_caption(full_caption: str) -> List[str]:
    """
    Extract text content from caption (quoted text).
    Enhanced to handle the model's specific output format
    
    Args:
        full_caption: The full image caption text
        
    Returns:
        List of text strings
    """
    text_elements = []
    
    # Primary pattern: quoted text after "contains the quoted text:"
    pattern1 = r'contains the quoted text:\s*"([^"]+)"'
    matches1 = re.findall(pattern1, full_caption)
    text_elements.extend([match.strip() for match in matches1])
    
    # Fallback pattern: all quoted text
    if not text_elements:
        pattern2 = r'"([^"]+)"'
        matches2 = re.findall(pattern2, full_caption)
        
        # Filter out layout references and keep substantial text
        for match in matches2:
            text = match.strip()
            if (len(text) > 5 and 
                not text.startswith('Layout ') and 
                not text.startswith('The image is') and
                'focused on' not in text):
                text_elements.append(text)
    
    return text_elements

def extract_figure_descriptions_from_caption(full_caption: str, n_figures: int) -> List[str]:
    """
    Extract figure descriptions from caption or generate appropriate ones
    """
    # Try to extract figure descriptions from context/content
    figure_descriptions = []
    
    # Basic figure descriptions based on common infographic elements
    base_descriptions = [
        "A visual chart or graph illustrating key data points",
        "An icon representing the main topic or theme",
        "A decorative element highlighting important information", 
        "A graphical representation supporting the narrative",
        "An illustrative diagram complementing the text",
        "A stylized symbol relevant to the content",
        "A visual component enhancing layout structure",
        "An informative graphic element",
        "A thematic illustration supporting key points",
        "A design element providing visual hierarchy"
    ]
    
    # Return appropriate number of descriptions
    for i in range(n_figures):
        if i < len(base_descriptions):
            figure_descriptions.append(base_descriptions[i])
        else:
            figure_descriptions.append(f"A visual element supporting the content ({i+1})")
    
    return figure_descriptions

def create_wiki_layout_from_new_format(
    full_image_caption: str,
    bbox_set: Dict,
    color_mapping: Dict[str, int],
    layouts: List[Dict],
    infographic_id: int,
    original_context: str = None,
    original_qa_pairs: List[Dict] = None,
    original_id: str = None,
    res_w: int = 896,
    res_h: int = 2240,
) -> Dict:
    """
    Create wiki layout structure from new format similar to merge_narrator_bboxes.py
    """
    bboxes = bbox_set.get('bboxes', [])
    
    # Extract text elements from the generated caption
    text_elements = extract_text_elements_from_caption(full_image_caption)
    figure_descriptions = extract_figure_descriptions_from_caption(full_image_caption, len([l for l in layouts if l['type'] == 'figure']))
    
    # Create output layers
    output_layers = []
    
    # 1. Base layer with full caption
    base_layer = {
        'category': 'base',
        'top_left': [0, 0],
        'bottom_right': [res_w, res_h],  
        'caption': full_image_caption
    }
    output_layers.append(base_layer)
    
    # 2. Background element layer (full canvas)
    background_element = {
        'category': 'element',
        'top_left': [0, 0],
        'bottom_right': [res_w, res_h],   # was [896, 2240]
        'caption': "Clean white background with professional layout"
    }
    output_layers.append(background_element)
    
    # 3. Figure element layers
    figure_layouts = [l for l in layouts if l['type'] == 'figure']
    for i, layout in enumerate(figure_layouts):
        if i < len(figure_descriptions):
            description = figure_descriptions[i]
        else:
            description = "A visual element supporting the content"
        
        element_layer = {
            'category': 'element',
            'top_left': [layout['bbox'][0], layout['bbox'][1]],
            'bottom_right': [layout['bbox'][2], layout['bbox'][3]],
            'caption': description
        }
        output_layers.append(element_layer)
    
    # 4. Text layers
    text_layouts = [l for l in layouts if l['type'] == 'text']
    default_font = 'en-font-67'  # Default font
    default_color = 'color-2'    # Default color
    
    for i, layout in enumerate(text_layouts):
        if i < len(text_elements):
            text_content = text_elements[i]
        else:
            text_content = f"Text content {i+1}"
        
        # Get color name if available
        color_name = layout.get('color', 'darkslategray')  # Default to darkslategray
        
        # Format caption similar to merge_narrator_bboxes.py
        caption = f'Text "{text_content}" in <{default_color}>, <{default_font}>. '
        
        text_layer = {
            'category': 'text',
            'top_left': [layout['bbox'][0], layout['bbox'][1]],
            'bottom_right': [layout['bbox'][2], layout['bbox'][3]],
            'caption': caption,
            'text': text_content
        }
        output_layers.append(text_layer)
    
    # Create final wiki result
    wiki_result = {
        'index': infographic_id,
        'layers_all': output_layers,
        'full_image_caption': full_image_caption,
        'original_bbox_index': bbox_set.get('index', -1),
        'original_context': original_context,
        'original_qa_pairs': original_qa_pairs or [],
        'original_id': original_id
    }
    
    return wiki_result

def process_sample_new_format(
    llm: LLM,
    item: Dict[str, Any],
    prompt_template_path: str,
    extracted_bboxes: List[Dict],
    color_mapping: Dict[str, int],
    infographic_id: int,
    min_text_area: int = 10000,
    res_w: int = 896,
    res_h: int = 2240,
) -> Optional[Dict]:
    """
    Process a single sample through the new format pipeline
    """
    try:
        # Read prompt template
        if not os.path.isfile(prompt_template_path):
            raise FileNotFoundError(f"Missing prompt template: {prompt_template_path}")
        
        prompt_template_text = read_text(prompt_template_path)
        
        # Extract data from item
        context = item.get('context', '').strip()
        qa_pairs = item.get('qa_pairs', [])
        
        if not context:
            raise ValueError("No context provided")
        
        # Select random layout with filtered text areas
        bbox_set = select_random_layout(extracted_bboxes, min_text_area)
        
        # Format layout for prompt
        background_caption, layouts, n_figures, m_texts = format_layout_for_prompt(
            bbox_set, 
            color_mapping
        )
        
        # Generate full image caption
        full_image_caption = generate_full_caption_new_format(
            llm,
            context,
            qa_pairs,
            background_caption,
            layouts,
            n_figures,
            m_texts,
            prompt_template_text,
            res_w=res_w,                
            res_h=res_h,                    
        )
        
        # Extract original information for tracking
        original_id = None
        if 'id' in item:
            original_id = item['id']
        elif qa_pairs and len(qa_pairs) > 0 and 'id' in qa_pairs[0]:
            original_id = qa_pairs[0]['id']
        else:
            original_id = f"context_{hash(context) % 1000000:06d}"
        
        # Create wiki layout from the new format
        wiki_layout = create_wiki_layout_from_new_format(
            full_image_caption,
            bbox_set,
            color_mapping,
            layouts,
            infographic_id,
            original_context=context,
            original_qa_pairs=qa_pairs,
            original_id=str(original_id),
            res_w=res_w,       
            res_h=res_h
        )
        
        return {
            'infographic_id': infographic_id,
            'wiki_layout': wiki_layout,
            'full_image_caption': full_image_caption,
            'background_caption': background_caption,
            'layouts': layouts,
            'n_figures': n_figures,
            'm_texts': m_texts,
            'bbox_set_index': bbox_set.get('index', -1),
            'original_context': context,
            'original_qa_pairs': qa_pairs,
            'original_id': str(original_id),
            'success': True,
            'min_text_area_used': min_text_area,
            # Debug information
            'debug': {
                'context_length': len(context),
                'qa_count': len(qa_pairs),
                'layout_count': len(layouts),
                'figures_count': n_figures,
                'texts_count': m_texts
            }
        }
        
    except Exception as e:
        print(f"Error processing sample {infographic_id}: {str(e)}")
        return {
            'infographic_id': infographic_id,
            'success': False,
            'error': str(e)
        }

def save_chunk_to_file_new_format(chunk: List[Optional[Dict]], output_dir: str, file_index: int) -> Optional[str]:
    """Save a chunk of results to file in wiki format like merge_narrator_bboxes.py"""
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
    none_info = f" ({none_count} failed)" if none_count > 0 else ""

    print(f"  ✓ Saved {len(wiki_layouts)} wiki layouts to {filename}{none_info}")
    if wiki_layouts:
        print(f"    Wiki IDs: {wiki_layouts[0]['index']}-{wiki_layouts[-1]['index']}")

    return filename


def main():
    """
    Main function for processing narrator data with new format
    """
    parser = argparse.ArgumentParser(description='Generate infographic captions using new format with context, QAs, background, and selected template')

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
                        help='Maximum tokens to generate')

    # Data / templates / IO
    parser.add_argument('--input_data', type=str,
                        default='/data/thangdd_workspace/InfographicDataPaper/info_squadv2/squad_v2_train.jsonl',
                        help='Path to Squad v2 JSONL file')
    parser.add_argument('--prompt_template', type=str,
                        default='./src/prompts/new_format_template.jinja',
                        help='Path to new format prompt template (Jinja2)')
    parser.add_argument('--extracted_bboxes', type=str,
                        default='./src/data/narrator/extracted_bboxes.json',
                        help='Path to extracted bboxes JSON file')
    parser.add_argument('--color_mapping', type=str,
                        default='./src/data/narrator/glyph/color_idx.json',
                        help='Path to color mapping JSON file')
    parser.add_argument('--output_dir', type=str,
                        default='./src/data/create_data/output/new_format',
                        help='Output directory for result files')

    # Control
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for processing (samples per file)')
    parser.add_argument('--start', type=int, default=1,
                        help='Start file index for data processing (inclusive, 1-based)')
    parser.add_argument('--end', type=int, default=None,
                        help='End file index for data processing (exclusive, 1-based)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (None for all)')
    parser.add_argument('--min_text_area', type=int, default=10000,
                        help='Minimum area (in pixels^2) for text bboxes to be included')
    parser.add_argument('--res_w', type=int, default=896,
                        help='Canvas/image width in pixels')
    parser.add_argument('--res_h', type=int, default=2240,
                        help='Canvas/image height in pixels')


    args = parser.parse_args()

    print("="*60)
    print("New Format Infographic Caption Generation")
    print("="*60)

    # Load input data
    print(f"\n[1/6] Loading input data from: {args.input_data}")
    input_data_full = load_squad_v2_data(args.input_data)
    print(f"Total unique contexts loaded: {len(input_data_full)}")

    # Calculate data slicing
    chunk_size = args.batch_size
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

    input_data_sliced = input_data_full[start_data_idx:end_data_idx]
    print(f"Sliced data: {len(input_data_sliced)} samples")

    if args.num_samples:
        input_data = input_data_sliced[:args.num_samples]
        print(f"Further limited to {args.num_samples} samples")
    else:
        input_data = input_data_sliced
        print(f"Processing all {len(input_data)} samples from slice")

    # Load extracted bboxes
    print(f"\n[2/6] Loading extracted bboxes from: {args.extracted_bboxes}")
    try:
        with open(args.extracted_bboxes, 'r', encoding='utf-8') as f:
            extracted_bboxes = json.load(f)
        print(f"  - Loaded {len(extracted_bboxes)} bbox sets")
    except Exception as e:
        print(f"Error loading extracted bboxes: {e}")
        return

    # Load color mapping
    print(f"\n[3/6] Loading color mapping from: {args.color_mapping}")
    try:
        color_mapping = load_color_mapping(args.color_mapping)
        print(f"  - Loaded {len(color_mapping)} color mappings")
        print(f"  - Min text area threshold: {args.min_text_area} pixels^2")
    except Exception as e:
        print(f"Error loading color mapping: {e}")
        return

    # Initialize LLM backend
    print(f"\n[4/6] Initializing LLM backend: {args.backend}")
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
    print(f"\n[5/6] Setting up output directory: {args.output_dir}")
    ensure_dir(args.output_dir)

    # Process data
    print(f"\n[6/6] Processing samples with new format")
    print(f"Prompt template: {args.prompt_template}")
    print("-"*60)

    results = []
    saved_files = []
    total_processed = 0
    successful_count = 0
    failed_count = 0

    for i, item in enumerate(tqdm(input_data, desc="Processing samples")):
        infographic_id = start_data_idx + i + 1

        result = process_sample_new_format(
            llm,
            item,
            args.prompt_template,
            extracted_bboxes,
            color_mapping,
            infographic_id,
            args.min_text_area,
            res_w=args.res_w,               
            res_h=args.res_h,           
        )

        if result and result.get("success", False):
            successful_count += 1
        else:
            failed_count += 1

        results.append(result)
        total_processed += 1

        # Save chunk when batch_size is reached
        if len(results) >= chunk_size:
            valid_results = [r for r in results if r is not None]
            if valid_results:
                first_infographic_id = valid_results[0]['infographic_id']
                file_index = (first_infographic_id - 1) // chunk_size + 1
            else:
                file_index = (start_data_idx + i - len(results) + 2) // chunk_size + 1

            filename = save_chunk_to_file_new_format(results, args.output_dir, file_index)
            if filename:
                saved_files.append(filename)
            results = []

    # Save final chunk if any
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

        filename = save_chunk_to_file_new_format(results, args.output_dir, file_index)
        if filename:
            saved_files.append(filename)

    # Final statistics
    print("\n" + "="*60)
    print("New Format Generation Complete - Final Statistics")
    print("="*60)

    if total_processed > 0:
        first_id = start_data_idx + 1
        last_id = start_data_idx + total_processed
        print(f"Infographic ID range: {first_id:06d} - {last_id:06d}")

    print(f"Total samples processed: {total_processed}")
    print(f"Total files saved: {len(saved_files)}")
    print(f"Successful: {successful_count} ({successful_count/total_processed*100:.1f}%)")
    print(f"Failed: {failed_count} ({failed_count/total_processed*100:.1f}%)")
    print(f"Min text area used: {args.min_text_area} pixels^2")
    print(f"Output directory: {args.output_dir}")

    print(f"\nFiles saved:")
    for filename in saved_files:
        print(f"  - {filename}")

    print("\n" + "="*60)
    print("New Format Generation complete!")
    print("="*60)


if __name__ == '__main__':
    main()