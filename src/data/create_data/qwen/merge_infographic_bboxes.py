import json
import random
from typing import List, Dict, Any, Tuple
import os
import argparse


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_infographic_files_from_directory(directory_path: str, start_wiki: int, end_wiki: int) -> List[Dict]:
    """
    Load infographic files from directory based on wiki ID range.
    
    Args:
        directory_path: Path to directory containing infographic*.json files
        start_wiki: Start wiki ID (inclusive)
        end_wiki: End wiki ID (exclusive)
    
    Returns:
        List of infographic data entries within the specified range
    """
    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_path} does not exist")
        return []
    
    infographic_data = []
    
    # Get all infographic*.json files in directory
    infographic_files = []
    for filename in os.listdir(directory_path):
        if filename.startswith('infographic') and filename.endswith('.json'):
            infographic_files.append(filename)
    
    # Sort files by number
    infographic_files.sort()
    
    print(f"Found {len(infographic_files)} infographic files in {directory_path}")
    
    # Load each file and filter by wiki ID range
    for filename in infographic_files:
        filepath = os.path.join(directory_path, filename)
        try:
            file_data = load_json(filepath)
            
            # Filter entries based on infographic_id within range
            for entry in file_data:
                if 'infographic_id' in entry:
                    infographic_id = entry['infographic_id']
                    if start_wiki < infographic_id <= end_wiki:  # Note: start_wiki < id <= end_wiki
                        infographic_data.append(entry)
            
            print(f"  Loaded {len(file_data)} entries from {filename}")
            
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue
    
    print(f"Total loaded infographic entries in range [{start_wiki+1}:{end_wiki}]: {len(infographic_data)}")
    return infographic_data


def save_json(data: Any, filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def calculate_bbox_area(bbox: Dict) -> int:
    """Calculate area of a bounding box."""
    top_left = bbox['top_left']
    bottom_right = bbox['bottom_right']
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    return width * height


def bboxes_overlap(bbox1: Dict, bbox2: Dict, threshold: float = 0.3) -> bool:
    """Check if two bounding boxes overlap significantly."""
    x1_min, y1_min = bbox1['top_left']
    x1_max, y1_max = bbox1['bottom_right']
    x2_min, y2_min = bbox2['top_left']
    x2_max, y2_max = bbox2['bottom_right']
    
    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = x_overlap * y_overlap
    
    # Calculate areas
    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)
    
    # Check if intersection is significant relative to smaller bbox
    min_area = min(area1, area2)
    return (intersection / min_area) > threshold if min_area > 0 else False


def select_largest_non_overlapping_bboxes(
    bboxes: List[Dict], 
    category: str, 
    count: int
) -> List[Dict]:
    """
    Select the largest non-overlapping bounding boxes of a specific category.
    
    Args:
        bboxes: List of all bboxes
        category: Category to filter ('element' or 'text')
        count: Number of bboxes to select
    
    Returns:
        List of selected bboxes
    """
    # Filter by category and exclude base
    filtered = [b for b in bboxes if b.get('category') == category]
    
    # Sort by area (largest first)
    filtered.sort(key=calculate_bbox_area, reverse=True)
    
    selected = []
    for bbox in filtered:
        # Check if this bbox overlaps with any already selected
        if not any(bboxes_overlap(bbox, s) for s in selected):
            selected.append(bbox)
            if len(selected) >= count:
                break
    
    return selected


def get_random_colors(color_idx: Dict, num_colors: int) -> List[str]:
    """Get random colors from color index."""
    colors = list(color_idx.keys())
    # Exclude pure white and black for better visibility
    colors = [c for c in colors if c not in ['white', 'black']]
    return random.sample(colors, min(num_colors, len(colors)))


def get_random_font(font_idx: Dict) -> str:
    """Get a random English font from font index."""
    # Always use English fonts only
    fonts = [k for k in font_idx.keys() if k.startswith('en-')]
    
    if not fonts:
        # Final fallback if no English fonts found
        fonts = list(font_idx.keys())
    
    return random.choice(fonts) if fonts else "en-YACgEQNAr7w,1"


def extract_text_from_caption(caption: str) -> str:
    """Extract text content from a caption string."""
    # Caption format: 'Text "content here"'
    if caption.startswith('Text "') or caption.startswith('Text \"'):
        # Find the quoted text
        start = caption.find('"')
        if start == -1:
            start = caption.find('\"')
        if start != -1:
            end = caption.find('"', start + 1)
            if end == -1:
                end = caption.find('\"', start + 1)
            if end != -1:
                return caption[start + 1:end]
    return ""


def find_background_bboxes_for_data(bboxes: List[Dict], count: int = 1) -> List[Dict]:
    """
    Find suitable background text bboxes for data placement (can overlap with other elements).
    
    Args:
        bboxes: List of available bboxes
        count: Number of bboxes to select for data placement
    
    Returns:
        List of selected background bboxes for data
    """
    # Filter text bboxes (data will be placed as text elements)
    text_bboxes = [b for b in bboxes if b.get('category') == 'text']
    
    # Sort by area (largest first) - prefer larger areas for data display
    text_bboxes.sort(key=calculate_bbox_area, reverse=True)
    
    # Return the largest text bboxes (no overlap checking needed since data goes to background)
    return text_bboxes[:count]


def split_data_caption(data_caption: str) -> List[str]:
    """
    Split data caption by semicolons and clean up the text.
    
    Args:
        data_caption: Data caption string with semicolon-separated items
    
    Returns:
        List of individual data items (max 10)
    """
    # Split by semicolon and clean up
    items = [item.strip() for item in data_caption.split(';') if item.strip()]
    
    # Limit to 10 items maximum
    return items[:10]


def create_data_text_bboxes(
    background_bbox: Dict, 
    data_items: List[str], 
    selected_colors: List[str], 
    color_idx: Dict, 
    font_idx: Dict,
    selected_font_id: Any
) -> List[Dict]:
    """
    Create individual text bboxes for data items within the background bbox.
    Data items will be placed as background elements that can overlap with other layers.
    
    Args:
        background_bbox: The background bbox to place data items in
        data_items: List of data text items
        selected_colors: Available colors
        color_idx: Color index mapping
        selected_font_id: Font ID to use
    
    Returns:
        List of text bbox layers for background data
    """
    if not data_items:
        return []
    
    x1, y1 = background_bbox['top_left']
    x2, y2 = background_bbox['bottom_right']
    
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    num_items = len(data_items)
    
    # Decide layout based on bbox aspect ratio and number of items
    is_wide = bbox_width > bbox_height
    
    if num_items <= 5:
        # Single row or column
        if is_wide:
            # Single row layout
            rows, cols = 1, num_items
        else:
            # Single column layout
            rows, cols = num_items, 1
    else:
        # Multiple rows/columns (max 2 rows/cols)
        if is_wide:
            # 2 rows layout
            rows = 2
            cols = (num_items + 1) // 2  # Ceiling division
        else:
            # 2 columns layout
            cols = 2
            rows = (num_items + 1) // 2  # Ceiling division
    
    # Calculate individual item dimensions
    item_width = bbox_width // cols
    item_height = bbox_height // rows
    
    text_bboxes = []
    
    for i, data_item in enumerate(data_items):
        # Calculate position
        row = i // cols
        col = i % cols
        
        item_x1 = x1 + col * item_width
        item_y1 = y1 + row * item_height
        item_x2 = item_x1 + item_width
        item_y2 = item_y1 + item_height
        
        # Select random color for this data item
        color = random.choice(selected_colors)
        color_id = color_idx[color]
        
        # Format caption similar to regular text (line 343 format)
        if selected_font_id is None:
            # Fallback font
            try:
                any_en_font_id = next(v for k, v in font_idx.items() if k.startswith('en-'))
            except StopIteration:
                any_en_font_id = 0
            font_token = f'en-font-{any_en_font_id}'
        else:
            font_token = f'en-font-{selected_font_id}'

        caption = f'Text "{data_item}" in <color-{color_id}>, <{font_token}>. '
        
        text_bbox = {
            'category': 'text',
            'top_left': [item_x1, item_y1],
            'bottom_right': [item_x2, item_y2],
            'caption': caption,
            'text': data_item
        }
        
        text_bboxes.append(text_bbox)
    
    return text_bboxes


def determine_language_prefix(index_str: str) -> str:
    """Determine language prefix from index string."""
    if isinstance(index_str, str):
        if index_str.startswith('cn_'):
            return 'cn'
        elif index_str.startswith('de_'):
            return 'de'
        elif index_str.startswith('es_'):
            return 'es'
        elif index_str.startswith('fr_'):
            return 'fr'
        elif index_str.startswith('it_'):
            return 'it'
        elif index_str.startswith('jp_'):
            return 'jp'
        elif index_str.startswith('kr_'):
            return 'kr'
        elif index_str.startswith('pt_'):
            return 'pt'
        elif index_str.startswith('ru_'):
            return 'ru'
    return 'en'


def merge_infographic_data(
    extracted_bboxes: List[Dict],
    infographic_generated: List[Dict],
    color_idx: Dict,
    font_idx: Dict,
    start_wiki_idx: int = 0,
    include_data: bool = False,
    include_timeline: bool = False
) -> List[Dict]:
    """
    Merge extracted bboxes with generated infographic data.
    
    Args:
        extracted_bboxes: List of bbox data from extracted_bboxes.json
        infographic_generated: List of infographic data from infographic_generated.json
        color_idx: Color index mapping
        font_idx: Font index mapping
        start_wiki_idx: Starting wiki index for generating unique IDs
        include_data: Whether to include data layers in the merged layout (default: False)
        include_timeline: Whether to include timeline layers in the merged layout (default: False)
    
    Returns:
        List of merged infographic data with unique wiki IDs
    """
    result = []
    
    # Create a mapping of indices from extracted_bboxes
    bbox_by_index = {item['index']: item for item in extracted_bboxes}
    available_indices = list(bbox_by_index.keys())
    
    for wiki_idx, infographic in enumerate(infographic_generated):
        # Handle both formats: direct infographic data or wrapped in generated_infographic
        if 'generated_infographic' in infographic:
            gen_info = infographic['generated_infographic']
        else:
            # Direct format (from directory files)
            gen_info = infographic
        
        # Skip if gen_info is not a dictionary (could be string or None)
        if not isinstance(gen_info, dict):
            print(f"Warning: Skipping infographic {wiki_idx} - gen_info is not a dictionary: {type(gen_info)}")
            continue
            
        # Skip if no layers
        if 'layers_all' not in gen_info or not gen_info['layers_all']:
            print(f"Warning: Skipping infographic {wiki_idx} - no layers_all found")
            continue
        # Generate unique wiki index based on start_wiki_idx
        wiki_id = start_wiki_idx + wiki_idx + 1  # Start from 1, not 0
        
        # Process all layers (no limit, typically 10-20)
        all_layers = gen_info['layers_all']
        
        # Separate layers by category
        figure_layers = [l for l in all_layers if l.get('category') == 'figure']
        text_layers = [l for l in all_layers if l.get('category') == 'text']
        data_layers = [l for l in all_layers if l.get('category') == 'data'] if include_data else []
        timeline_layers = [l for l in all_layers if l.get('category') == 'timeline'] if include_timeline else []
        
        # Check if we should skip data/timeline layers based on count
        text_figure_count = len(text_layers) + len(figure_layers)
        if text_figure_count > 20:
            # Skip data and timeline layers regardless of user preference
            data_layers = []
            timeline_layers = []
            print(f"Skipping data/timeline layers for wiki {wiki_id:06d}: text+figure count ({text_figure_count}) > 20")
        elif not include_data and not include_timeline:
            print(f"Skipping data/timeline layers for wiki {wiki_id:06d}: disabled by user (recommended for better image quality)")
        elif include_data and data_layers:
            print(f"Including {len(data_layers)} data layers for wiki {wiki_id:06d} (may affect image quality)")
        elif include_timeline and timeline_layers:
            print(f"Including {len(timeline_layers)} timeline layers for wiki {wiki_id:06d} (may affect image quality)")
        
        # Reconstruct layers list for processing
        layers = figure_layers + text_layers + data_layers + timeline_layers

        
        # Select a random bbox index from available indices
        if not available_indices:
            print(f"Warning: No more available bbox indices for wiki {wiki_id:06d}")
            continue
        
        selected_bbox_index = random.choice(available_indices)
        # Remove the selected index to avoid reuse
        available_indices.remove(selected_bbox_index)
        
        bbox_data = bbox_by_index[selected_bbox_index]
        bboxes = bbox_data['bboxes']
        
        # Select colors and font for this infographic (English only)
        num_colors = random.randint(1, 4)
        selected_colors = get_random_colors(color_idx, num_colors)
        selected_font = get_random_font(font_idx)
        selected_font_id = font_idx.get(selected_font, None)
        
        # Build layers_all for output
        output_layers = []
        
        # First, add the base layer (full_image_caption)
        base_bbox = next((b for b in bboxes if b.get('category') == 'base'), None)
        if base_bbox:
            base_layer = {
                'category': 'base',
                'top_left': base_bbox['top_left'],
                'bottom_right': base_bbox['bottom_right'],
                'caption': gen_info.get('full_image_caption', '')
            }
            output_layers.append(base_layer)
        
        # Count how many of each category we need
        figure_count = sum(1 for layer in layers if layer.get('category') == 'figure')
        text_count = sum(1 for layer in layers if layer.get('category') == 'text')
        
        # Separate full image elements and regular elements
        element_bboxes = [b for b in bboxes if b.get('category') == 'element']
        full_image_elements = [b for b in element_bboxes if b['bottom_right'] == [896, 2240]]
        regular_elements = [b for b in element_bboxes if b['bottom_right'] != [896, 2240]]
        
        # Strategy: Add background elements (full image) + decorative elements (smaller)
        # 1. Always add 1-2 full image elements for background
        num_full_image = min(len(full_image_elements), max(1, 2))
        selected_figure_bboxes = full_image_elements[:num_full_image]
        
        # 2. Add decorative elements (smaller, non-overlapping with each other, but can overlap with full image)
        # Add at least 3-5 decorative elements to match original infographic style
        num_decorative = random.randint(3, 5)
        
        # Sort regular elements by area and select largest non-overlapping
        regular_elements.sort(key=calculate_bbox_area, reverse=True)
        decorative_count = 0
        selected_decorative = []
        
        for bbox in regular_elements:
            # Only check overlap with other decorative elements, not with full image backgrounds
            overlaps = any(bboxes_overlap(bbox, s) for s in selected_decorative)
            if not overlaps:
                selected_figure_bboxes.append(bbox)
                selected_decorative.append(bbox)
                decorative_count += 1
                if decorative_count >= num_decorative:
                    break
        
        # Select bboxes for text
        selected_text_bboxes = select_largest_non_overlapping_bboxes(
            bboxes, 'text', text_count
        )
        
        # Add all selected element bboxes
        # First element(s) will be background (full image), rest are decorations
        for idx, bbox in enumerate(selected_figure_bboxes):
            # Use figure layer caption if available, otherwise use generic caption
            if idx < len([l for l in layers if l.get('category') == 'figure']):
                # Get the corresponding figure layer
                figure_layers = [l for l in layers if l.get('category') == 'figure']
                layer_caption = figure_layers[idx].get('caption', '')
            else:
                # Generic caption for decorative elements
                is_full_image = bbox['bottom_right'] == [896, 2240]
                if is_full_image:
                    layer_caption = "A background layer filling the entire canvas."
                else:
                    layer_caption = "A decorative element or icon."
            
            output_layer = {
                'category': 'element',
                'top_left': bbox['top_left'],
                'bottom_right': bbox['bottom_right'],
                'caption': layer_caption
            }
            output_layers.append(output_layer)
        
        # Add all text bboxes
        text_layers = [l for l in layers if l.get('category') == 'text']
        for idx, bbox in enumerate(selected_text_bboxes):
            if idx < len(text_layers):
                layer_caption = text_layers[idx].get('caption', '')
                # Extract text content from caption
                text_content = extract_text_from_caption(layer_caption)
            else:
                # Should not happen, but handle gracefully
                text_content = ""
            
            # Select random color for this text
            color = random.choice(selected_colors)
            color_id = color_idx[color]
            
            # Format caption with color and font (expected: <color-{id}>, <en-font-{id}>)
            if selected_font_id is None:
                # Fallback: try to map any en- font id; else default to 0
                try:
                    any_en_font_id = next(v for k, v in font_idx.items() if k.startswith('en-'))
                except StopIteration:
                    any_en_font_id = 0
                font_token = f'en-font-{any_en_font_id}'
            else:
                font_token = f'en-font-{selected_font_id}'

            caption = f'Text "{text_content}" in <color-{color_id}>, <{font_token}>. '
            
            output_layer = {
                'category': 'text',
                'top_left': bbox['top_left'],
                'bottom_right': bbox['bottom_right'],
                'caption': caption,
                'text': text_content
            }
            output_layers.append(output_layer)
        
        # Process data layers if they exist and are enabled (place them at the beginning as background)
        data_layers_filtered = [l for l in layers if l.get('category') == 'data']
        if data_layers_filtered and include_data:
            # Find background bboxes for data placement (can overlap with other elements)
            data_background_bboxes = find_background_bboxes_for_data(bboxes, count=len(data_layers_filtered))
            
            if data_background_bboxes:
                # Process each data layer
                for data_idx, data_layer in enumerate(data_layers_filtered):
                    if data_idx >= len(data_background_bboxes):
                        break  # No more background bboxes available
                    
                    background_bbox = data_background_bboxes[data_idx]
                    data_caption = data_layer.get('caption', '')
                    
                    # Split data caption by semicolons
                    data_items = split_data_caption(data_caption)
                    
                    if data_items:
                        # Create individual text bboxes for data items
                        data_text_bboxes = create_data_text_bboxes(
                            background_bbox,
                            data_items,
                            selected_colors,
                            color_idx,
                            font_idx,
                            selected_font_id
                        )
                        
                        # Insert data text bboxes at the beginning (background layer)
                        # This ensures they render behind other elements
                        for i, data_bbox in enumerate(data_text_bboxes):
                            output_layers.insert(i + 1, data_bbox)  # Insert after base layer
                        
                        print(f"Added {len(data_text_bboxes)} background data text bboxes for wiki {wiki_id:06d}")
            else:
                print(f"No suitable background bboxes found for data placement in wiki {wiki_id:06d}")
        
        # Process timeline layers if they exist and are enabled
        # Note: Timeline processing is not fully implemented yet - placeholder for future development
        timeline_layers_filtered = [l for l in layers if l.get('category') == 'timeline']
        if timeline_layers_filtered and include_timeline:
            print(f"Timeline layer processing is not yet implemented for wiki {wiki_id:06d}")
            # TODO: Implement timeline layer processing if needed in the future
        
        # Create the final structure with unique wiki ID
        result_item = {
            'index': wiki_id,
            'layers_all': output_layers,
            'full_image_caption': gen_info.get('full_image_caption', ''),
            'original_bbox_index': selected_bbox_index,  # Keep track of original bbox for debugging
            'original_infographic_id': infographic.get('id', '')
        }
        
        result.append(result_item)
    
    return result


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Merge extracted_bboxes.json with infographic data (from file or directory)'
    )
    parser.add_argument(
        '--extracted-bboxes',
        type=str,
        default=None,
        help='Path to extracted_bboxes.json (default: ./extracted_bboxes.json)'
    )
    parser.add_argument(
        '--infographic-generated',
        type=str,
        default=None,
        help='Path to infographic_generated.json or directory containing infographic files (default: auto-detect)'
    )
    parser.add_argument(
        '--infographic-dir',
        type=str,
        default=None,
        help='Directory containing infographic*.json files (default: src/data/create_data/output/infographic)'
    )
    parser.add_argument(
        '--color-idx',
        type=str,
        default=None,
        help='Path to color_idx.json (default: ../bizgen/glyph/color_idx.json)'
    )
    parser.add_argument(
        '--font-idx',
        type=str,
        default=None,
        help='Path to font_uni_10-lang_idx.json (default: ../bizgen/glyph/font_uni_10-lang_idx.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: ./merged_infographics.json)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--start-wiki',
        type=int,
        default=0,
        help='Start index for infographic generation (inclusive)'
    )
    parser.add_argument(
        '--end-wiki',
        type=int,
        default=None,
        help='End index for infographic generation (exclusive)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for wiki files (default: src/data/create_data/output/bizgen_format)'
    )
    parser.add_argument(
        '--layout_data',
        action='store_true',
        default=False,
        help='Include data layers in the merged layout (default: False, as it may reduce image quality)'
    )
    parser.add_argument(
        '--layout_timeline',
        action='store_true', 
        default=False,
        help='Include timeline layers in the merged layout (default: False, as it may reduce image quality)'
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths
    extracted_bboxes_path = args.extracted_bboxes or os.path.join(script_dir, 'extracted_bboxes.json')
    color_idx_path = args.color_idx or os.path.join(script_dir, '../bizgen/glyph/color_idx.json')
    font_idx_path = args.font_idx or os.path.join(script_dir, '../bizgen/glyph/font_uni_10-lang_idx.json')
    
    # Determine infographic data source
    if args.infographic_dir:
        infographic_dir = args.infographic_dir
    elif args.infographic_generated and os.path.isdir(args.infographic_generated):
        infographic_dir = args.infographic_generated
    else:
        # Default to output/infographic directory
        repo_root = os.path.abspath(os.path.join(script_dir, '../../../../'))
        infographic_dir = os.path.join(repo_root, 'src/data/create_data/output/infographic')
    
    # Fallback to single file if specified and exists
    infographic_file_path = None
    if args.infographic_generated and os.path.isfile(args.infographic_generated):
        infographic_file_path = args.infographic_generated
    elif not args.infographic_dir and not (args.infographic_generated and os.path.isdir(args.infographic_generated)):
        # Default single file path
        infographic_file_path = args.infographic_generated or os.path.join(script_dir, 'infographic_generated.json')
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to src/data/create_data/output/bizgen_format from repository root
        repo_root = os.path.abspath(os.path.join(script_dir, '../../../../'))
        output_dir = os.path.join(repo_root, 'src/data/create_data/output/bizgen_format')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    print(f"  - Bboxes: {extracted_bboxes_path}")
    print(f"  - Colors: {color_idx_path}")
    print(f"  - Fonts: {font_idx_path}")
    
    extracted_bboxes = load_json(extracted_bboxes_path)
    color_idx = load_json(color_idx_path)
    font_idx = load_json(font_idx_path)
    
    # Load infographic data
    end_idx = args.end_wiki if args.end_wiki is not None else (args.start_wiki + 50000)  # Default large range
    
    if infographic_file_path and os.path.exists(infographic_file_path):
        # Load from single file
        print(f"  - Infographics: {infographic_file_path} (single file)")
        infographic_generated_full = load_json(infographic_file_path)
        infographic_generated = infographic_generated_full[args.start_wiki:end_idx]
        print(f"Loaded {len(infographic_generated_full)} total infographic entries from single file")
        print(f"Processing slice [{args.start_wiki}:{end_idx}] = {len(infographic_generated)} entries")
    else:
        # Load from directory
        print(f"  - Infographics: {infographic_dir} (directory)")
        infographic_generated = load_infographic_files_from_directory(
            infographic_dir, 
            args.start_wiki, 
            end_idx
        )
    
    print(f"\nLoaded {len(extracted_bboxes)} bbox entries")
    print(f"Loaded {len(color_idx)} colors")
    print(f"Loaded {len(font_idx)} fonts")
    
    # Merge data
    print("\nMerging data...")
    print(f"Layout options: data={args.layout_data}, timeline={args.layout_timeline}")
    if not args.layout_data and not args.layout_timeline:
        print("Note: Data and timeline layers are disabled by default for better image quality.")
    merged_data = merge_infographic_data(
        extracted_bboxes,
        infographic_generated,
        color_idx,
        font_idx,
        start_wiki_idx=args.start_wiki,
        include_data=args.layout_data,
        include_timeline=args.layout_timeline
    )
    
    print(f"Generated {len(merged_data)} merged infographics")
    
    # Save results in chunks of 50 per file
    print(f"\nSaving to {output_dir}...")
    chunk_size = 50
    saved_files = []
    
    for i in range(0, len(merged_data), chunk_size):
        chunk = merged_data[i:i + chunk_size]
        
        # Calculate file index based on first wiki ID in chunk
        first_wiki_id = chunk[0]['index']
        file_index = (first_wiki_id - 1) // chunk_size + 1  # Convert to 1-based file indexing
        
        filename = f"wiki{file_index:06d}.json"
        filepath = os.path.join(output_dir, filename)
        
        save_json(chunk, filepath)
        saved_files.append(filename)
        
        print(f"  Saved {len(chunk)} infographics to {filename} (wiki IDs: {chunk[0]['index']}-{chunk[-1]['index']})")
    
    print(f"\nSaved {len(saved_files)} files total")
    print("Done!")
    
    # Print summary statistics
    print("\n=== Summary ===")
    total_layers = sum(len(item['layers_all']) for item in merged_data)
    total_text_layers = sum(
        sum(1 for layer in item['layers_all'] if layer.get('category') == 'text')
        for item in merged_data
    )
    total_element_layers = sum(
        sum(1 for layer in item['layers_all'] if layer.get('category') == 'element')
        for item in merged_data
    )
    total_base_layers = sum(
        sum(1 for layer in item['layers_all'] if layer.get('category') == 'base')
        for item in merged_data
    )
    
    if merged_data:
        first_wiki_id = merged_data[0]['index']
        last_wiki_id = merged_data[-1]['index']
        print(f"Wiki ID range: {first_wiki_id:06d} - {last_wiki_id:06d}")
    
    print(f"Total infographics: {len(merged_data)}")
    print(f"Total files saved: {len(saved_files)}")
    print(f"Total layers: {total_layers}")
    print(f"  - Base layers: {total_base_layers}")
    print(f"  - Element layers: {total_element_layers}")
    print(f"  - Text layers: {total_text_layers}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()

