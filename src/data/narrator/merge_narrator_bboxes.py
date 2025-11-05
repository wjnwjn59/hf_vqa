from typing import List, Dict, Any, Tuple
import json
import os
import argparse
import random
import re


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_squad_jsonl(filepath: str) -> Dict[str, Dict]:
    """
    Load SQuAD v2 JSONL file and create a mapping from ID to QA pairs.
    
    Args:
        filepath: Path to squad_v2_train.jsonl file
        
    Returns:
        Dictionary mapping question ID to QA data
    """
    id_to_qa = {}
    
    print(f"Loading SQuAD data from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                qa_id = data.get('id')
                if qa_id:
                    id_to_qa[qa_id] = {
                        'question': data.get('question', ''),
                        'answers': data.get('answers', {'text': [], 'answer_start': []}),
                        'id': qa_id
                    }
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line {line_num}")
                continue
    
    print(f"Loaded {len(id_to_qa)} QA pairs from SQuAD dataset")
    return id_to_qa


def save_json(data: Any, filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_infographic_files_from_directory(directory_path: str, start_file_idx: int, end_file_idx: int) -> List[Dict]:
    """
    Load infographic files from directory based on file index range.
    Each file contains 50 entries with infographic_id starting from (file_index-1)*50 + 1
    
    Args:
        directory_path: Path to directory containing infographic*.json files
        start_file_idx: Start file index (inclusive, 1-based)
        end_file_idx: End file index (exclusive, 1-based)
    
    Returns:
        List of infographic data entries from the specified files
    """
    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_path} does not exist")
        return []
    
    infographic_data = []
    
    print(f"Loading files infographic{start_file_idx:06d}.json to infographic{end_file_idx-1:06d}.json")
    
    for file_index in range(start_file_idx, end_file_idx):
        filename = f"infographic{file_index:06d}.json"
        filepath = os.path.join(directory_path, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: File {filename} does not exist, skipping")
            continue
            
        try:
            data = load_json(filepath)
            infographic_data.extend(data)
            print(f"  Loaded {len(data)} entries from {filename}")
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    print(f"Total loaded infographic entries from {end_file_idx - start_file_idx} files: {len(infographic_data)}")
    return infographic_data


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
    count: int,
    min_area: int = 0
) -> List[Dict]:
    """
    Select the largest non-overlapping bounding boxes of a specific category.
    
    Args:
        bboxes: List of all bboxes
        category: Category to filter ('element' or 'text')
        count: Number of bboxes to select
        min_area: Minimum area requirement for bboxes (pixels)
    
    Returns:
        List of selected bboxes
    """
    # Filter by category and exclude base
    filtered = [b for b in bboxes if b.get('category') == category]
    
    # For text category, apply minimum area filter
    if category == 'text' and min_area > 0:
        filtered = [b for b in filtered if calculate_bbox_area(b) >= min_area]
    
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
    # Exclude white for better visibility (keep black)
    colors = [c for c in colors if c not in ['white']]
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
            start = caption.find('"')
        if start != -1:
            end = caption.find('"', start + 1)
            if end == -1:
                end = caption.find('"', start + 1)
            if end != -1:
                return caption[start + 1:end]
    return ""


def extract_images_from_caption(full_caption: str) -> List[Dict]:
    """
    Extract image descriptions from caption using new format.
    Format: "description" (figure) or "description" (figure). or "description" (figure),
    
    Args:
        full_caption: The full image caption text
        
    Returns:
        List of dicts with 'description' key
    """
    image_elements = []
    
    # Pattern to match "content" (figure) with optional punctuation after
    # Matches: (figure), (figure). (figure), (figure); etc.
    pattern = r'"([^"]+)"\s*\(figure\)[.,;:!?]?'
    matches = re.findall(pattern, full_caption, re.IGNORECASE)
    
    for description in matches:
        description = description.strip()
        
        # Check if description already starts with proper prefix
        description_lower = description.lower()
        figure_prefixes = [
            'a visual', 'an illustration', 'the picture', 'a picture', 'an image',
            'the image', 'a graphic', 'the graphic', 'a diagram', 'the diagram',
            'an icon', 'the icon', 'a chart', 'the chart', 'a figure', 'the figure',
            'a photo', 'the photo', 'a drawing', 'the drawing', 'a representation',
            'the representation', 'a silhouette', 'the silhouette', 'a composition',
            'the composition', 'a design', 'the design', 'a cover', 'the cover',
            'a map', 'the map', 'a stylized'
        ]
        
        has_prefix = any(description_lower.startswith(prefix) for prefix in figure_prefixes)
        
        # If no prefix found, add a generic one
        if not has_prefix:
            description = f"An illustration of {description}"
        
        image_elements.append({
            'description': description
        })
    
    return image_elements


def extract_text_elements(full_caption: str) -> List[str]:
    """
    Extract text content from caption (quoted text with (text) tag).
    Format: "text content" (text) or "text content" (text). or "text content" (text),
    
    Args:
        full_caption: The full image caption text
        
    Returns:
        List of text strings
    """
    text_elements = []
    
    # Pattern to match "content" (text) with optional punctuation after
    # Matches: (text), (text). (text), (text); etc.
    pattern = r'"([^"]+)"\s*\(text\)[.,;:!?]?'
    matches = re.findall(pattern, full_caption, re.IGNORECASE)
    
    for text_content in matches:
        text_elements.append(text_content.strip())
    
    return text_elements


def clean_caption_text(caption: str) -> str:
    """
    Clean up caption text - remove (text) and (figure) tags along with their quoted content.
    This creates a clean narrative without the tagged elements.
    Handles tags with optional punctuation: (figure). (figure), (text). (text), etc.
    
    Args:
        caption: The full image caption with tags
        
    Returns:
        Cleaned caption text
    """
    # Remove "content" (figure) with optional punctuation - remove both quotes and tag
    caption = re.sub(r'"[^"]+"\s*\(figure\)[.,;:!?]?', '', caption, flags=re.IGNORECASE)
    
    # Remove (text) tag with optional punctuation but keep the quoted text content
    caption = re.sub(r'\(text\)[.,;:!?]?', '', caption, flags=re.IGNORECASE)
    
    # Clean up extra whitespace and normalize spacing
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    # Clean up multiple spaces around punctuation
    caption = re.sub(r'\s+([.,;:!?])', r'\1', caption)
    
    return caption


def extract_font_color_from_bboxes(bboxes: List[Dict], font_idx: Dict) -> Tuple[str, List[str]]:
    """
    Extract font and colors from text bboxes in the layout.
    Returns a tuple of (font_token, list_of_color_names).
    If no English font is found, returns a random English font.
    
    Args:
        bboxes: List of bbox data
        font_idx: Font index mapping
        
    Returns:
        Tuple of (font_token like 'en-font-1', list of color names)
    """
    text_bboxes = [b for b in bboxes if b.get('category') == 'text']
    
    # Extract font and color info from text bboxes
    fonts_found = []
    colors_found = []
    
    for bbox in text_bboxes:
        font_color_info = bbox.get('font_color_info', '')
        if font_color_info:
            # Extract font: pattern <en-font-X> or <cn-font-X>, etc.
            font_match = re.search(r'<(en-font-\d+)>', font_color_info)
            if font_match:
                fonts_found.append(font_match.group(1))
            
            # Extract color: pattern <color-X>
            color_match = re.search(r'<color-(\d+)>', font_color_info)
            if color_match:
                colors_found.append(int(color_match.group(1)))
    
    # Determine font to use (should be consistent across layout)
    font_token = None
    if fonts_found:
        # Use the most common English font found
        font_token = max(set(fonts_found), key=fonts_found.count)
    else:
        # Fallback: random English font
        en_fonts = [k for k in font_idx.keys() if k.startswith('en-')]
        if en_fonts:
            selected_font = random.choice(en_fonts)
            font_id = font_idx[selected_font]
            font_token = f'en-font-{font_id}'
        else:
            font_token = 'en-font-0'
    
    # Get unique colors and convert to color names
    unique_color_ids = list(set(colors_found))
    
    return font_token, unique_color_ids


def get_color_name_from_id(color_id: int, color_idx: Dict) -> str:
    """Get color name from color ID."""
    for name, cid in color_idx.items():
        if cid == color_id:
            return name
    return 'black'  # fallback


def is_white_or_light_color(color_name: str) -> bool:
    """
    Check if a color is white or very light (near-white).
    These colors are hard to read on white/light backgrounds.
    
    Args:
        color_name: Name of the color
        
    Returns:
        True if color is white or near-white
    """
    # List of white and near-white colors that should be replaced
    white_like_colors = {
        'white', 'whitesmoke', 'snow', 'ghostwhite', 'floralwhite',
        'linen', 'oldlace', 'ivory', 'seashell', 'beige',
        'cornsilk', 'lavenderblush', 'mistyrose', 'papayawhip',
        'blanchedalmond', 'bisque', 'antiquewhite', 'lemonchiffon',
        'lightgoldenrodyellow', 'lightyellow', 'honeydew', 'mintcream',
        'azure', 'aliceblue', 'lavender', 'lightcyan', 'gainsboro'
    }
    
    return color_name.lower() in white_like_colors


def replace_white_color_with_black(color_name: str, color_idx: Dict) -> str:
    """
    Replace white or near-white colors with black for better readability.
    
    Args:
        color_name: Original color name
        color_idx: Color index mapping
        
    Returns:
        Color name (black if original was white-like, otherwise unchanged)
    """
    if is_white_or_light_color(color_name):
        return 'black'
    return color_name


def calculate_bbox_center_y(bbox: Dict) -> float:
    """Calculate the vertical center of a bounding box."""
    return (bbox['top_left'][1] + bbox['bottom_right'][1]) / 2


def calculate_canvas_coverage_score(selected_bboxes: List[Dict], canvas_height: int = 2240) -> float:
    """
    Calculate how well the selected bboxes cover the entire canvas vertically.
    Returns a score from 0 to 1, where 1 means perfect coverage.
    """
    if not selected_bboxes:
        return 0.0
    
    # Divide canvas into regions and check coverage
    num_regions = 4  # Top, upper-mid, lower-mid, bottom
    region_height = canvas_height // num_regions
    region_coverage = [False] * num_regions
    
    for bbox in selected_bboxes:
        center_y = calculate_bbox_center_y(bbox)
        region_idx = min(int(center_y // region_height), num_regions - 1)
        region_coverage[region_idx] = True
    
    return sum(region_coverage) / num_regions


def select_spatially_distributed_bboxes(
    bboxes: List[Dict], 
    target_count: int, 
    canvas_height: int = 2240
) -> List[Dict]:
    """
    Select bboxes that provide good spatial distribution across the canvas
    while prioritizing larger bboxes.
    
    Args:
        bboxes: List of available bboxes
        target_count: Number of bboxes to select
        canvas_height: Height of the canvas for spatial calculation
    
    Returns:
        List of selected bboxes with good spatial distribution
    """
    if not bboxes or target_count <= 0:
        return []
    
    if len(bboxes) <= target_count:
        return bboxes
    
    # Sort by area (largest first) as base priority
    sorted_bboxes = sorted(bboxes, key=calculate_bbox_area, reverse=True)
    
    selected = []
    
    # Greedy selection with spatial awareness
    for candidate in sorted_bboxes:
        if len(selected) >= target_count:
            break
            
        # Check overlap with already selected bboxes
        overlaps_with_selected = any(bboxes_overlap(candidate, s) for s in selected)
        
        if not overlaps_with_selected:
            # Calculate coverage improvement
            coverage_before = calculate_canvas_coverage_score(selected, canvas_height)
            coverage_after = calculate_canvas_coverage_score(selected + [candidate], canvas_height)
            coverage_improvement = coverage_after - coverage_before
            
            # Area-based priority score
            area_score = calculate_bbox_area(candidate) / max(calculate_bbox_area(b) for b in sorted_bboxes)
            
            # Combined score: prioritize area but boost coverage improvement
            combined_score = area_score + (coverage_improvement * 2.0)
            
            # Accept if it's beneficial or we need more bboxes
            if coverage_improvement > 0 or len(selected) < target_count // 2:
                selected.append(candidate)
    
    # If we still don't have enough, fill with largest remaining non-overlapping bboxes
    if len(selected) < target_count:
        for candidate in sorted_bboxes:
            if len(selected) >= target_count:
                break
            if candidate not in selected:
                overlaps_with_selected = any(bboxes_overlap(candidate, s) for s in selected)
                if not overlaps_with_selected:
                    selected.append(candidate)
    
    print(f"    Spatial distribution: {len(selected)}/{target_count} bboxes, coverage score: {calculate_canvas_coverage_score(selected, canvas_height):.2f}")
    
    return selected[:target_count]


def find_suitable_layouts_for_content(
    image_count: int,
    text_count: int,
    extracted_bboxes: List[Dict],
    min_text_area: int = 10000
) -> List[Dict]:
    """
    Find all suitable layouts that can accommodate the required number of images and text elements.
    Returns a list of suitable layouts sorted by score.
    Note: Title (first text) uses largest text bbox, other texts need area >= min_text_area.
    
    Args:
        image_count: Number of image elements needed
        text_count: Number of text elements needed
        extracted_bboxes: List of all available layouts
        min_text_area: Minimum area for non-title text bboxes (default 10000 pixels²)
    
    Returns:
        List of suitable layout dicts sorted by score (best first)
    """
    suitable_layouts = []
    
    print(f"  Searching for layouts: need {image_count} images, {text_count} texts (title: largest bbox, others: >= {min_text_area} px²)")
    
    for bbox_data in extracted_bboxes:
        bboxes = bbox_data['bboxes']
        
        # Count available element bboxes (excluding background)
        element_bboxes = [b for b in bboxes if b.get('category') == 'element']
        regular_elements = [b for b in element_bboxes if b['bottom_right'] != [896, 2240]]
        
        # Count available text bboxes with new logic
        text_bboxes = [b for b in bboxes if b.get('category') == 'text']
        
        # For capacity calculation, simulate the new selection logic
        if text_count > 0:
            # Title uses largest text bbox
            title_available = select_largest_non_overlapping_bboxes(bboxes, 'text', 1, min_area=0)
            
            # Other texts need min_area threshold
            if text_count > 1:
                other_texts_available = select_largest_non_overlapping_bboxes(bboxes, 'text', text_count, min_area=min_text_area)
                # Remove title bbox if it overlaps
                if title_available and other_texts_available:
                    title_bbox = title_available[0]
                    other_texts_available = [bbox for bbox in other_texts_available 
                                           if not (bbox['top_left'] == title_bbox['top_left'] and 
                                                 bbox['bottom_right'] == title_bbox['bottom_right'])]
                text_capacity = min(1 + len(other_texts_available), text_count)
            else:
                text_capacity = len(title_available)
        else:
            text_capacity = 0
        
        # Apply non-overlapping selection for elements
        available_elements = select_largest_non_overlapping_bboxes(bboxes, 'element', image_count + 10)  # Extra margin
        
        element_capacity = len(available_elements)
        
        # Calculate score based on how well this layout matches requirements
        element_score = min(element_capacity, image_count) / max(image_count, 1)
        text_score = min(text_capacity, text_count) / max(text_count, 1)
        
        # Bonus for exact match or having more capacity than needed
        element_bonus = 1.0 if element_capacity >= image_count else 0.0
        text_bonus = 1.0 if text_capacity >= text_count else 0.0
        
        # Combined score (weighted towards text since it has area constraints)
        total_score = (element_score * 0.4 + text_score * 0.6) + (element_bonus * 0.2 + text_bonus * 0.3)
        
        # Only include layouts that can meet the requirements (score >= 1.0 means it can fit)
        if element_capacity >= image_count and text_capacity >= text_count:
            suitable_layouts.append({
                'layout': bbox_data,
                'score': total_score,
                'element_capacity': element_capacity,
                'text_capacity': text_capacity
            })
    
    # Sort by score (best first)
    suitable_layouts.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"  Found {len(suitable_layouts)} suitable layouts")
    
    return suitable_layouts


def merge_narrator_data(
    infographic_generated: List[Dict],
    extracted_bboxes: List[Dict],
    color_idx: Dict,
    font_idx: Dict,
    squad_id_to_qa: Dict[str, Dict],
    start_wiki_idx: int = 0,
    use_infographic_id: bool = False
) -> List[Dict]:
    """
    Process narrator-generated infographic data and merge with bboxes.
    Uses round-robin with shuffling to distribute layouts evenly.
    
    Args:
        infographic_generated: List of infographic data with full_image_caption and background_caption
        extracted_bboxes: List of bbox data from extracted_bboxes.json
        color_idx: Color index mapping
        font_idx: Font index mapping
        squad_id_to_qa: Mapping from question ID to QA pairs
        start_wiki_idx: Starting wiki index for generating unique IDs (used when use_infographic_id=False)
        use_infographic_id: If True, use infographic_id from each entry instead of calculating from start_wiki_idx
    
    Returns:
        List of merged infographic data (full layout version only)
    """
    result = []
    
    # Statistics tracking
    stats = {
        'total_infographics': len(infographic_generated),
        'successful': 0,
        'skipped': 0,
        'fallback_used': 0,
        'skipped_wiki_ids': []
    }
    
    # Group infographics by their content requirements for better layout distribution
    # Create a shuffled list of suitable layouts for each infographic
    layout_assignments = []
    
    # First pass: find suitable layouts for each infographic
    print("\n=== Phase 1: Finding suitable layouts for each infographic ===")
    for wiki_idx, infographic in enumerate(infographic_generated):
        # Determine wiki_id based on mode
        if use_infographic_id:
            # Single file mode: use infographic_id from the entry
            wiki_id = infographic.get('infographic_id')
            if wiki_id is None:
                print(f"Warning: infographic at index {wiki_idx} missing 'infographic_id', skipping")
                continue
        else:
            # Directory mode: calculate from start_wiki_idx
            wiki_id = start_wiki_idx + wiki_idx + 1
        
        # Get the caption data
        generated_infographic = infographic.get('generated_infographic', '')
        
        if isinstance(generated_infographic, str):
            full_image_caption = generated_infographic
        elif isinstance(generated_infographic, dict):
            full_image_caption = generated_infographic.get('full_image_caption', '')
        else:
            print(f"Warning: Unexpected generated_infographic type for wiki {wiki_id}")
            continue
        
        if not full_image_caption:
            print(f"Warning: No full_image_caption for wiki {wiki_id}, skipping")
            continue
        
        # Extract image and text elements
        image_elements = extract_images_from_caption(full_image_caption)
        text_elements = extract_text_elements(full_image_caption)
        
        print(f"\nInfographic {wiki_idx + 1}/{len(infographic_generated)} (wiki {wiki_id}):")
        print(f"  Found {len(image_elements)} image elements, {len(text_elements)} text elements")
        
        # Find all suitable layouts
        suitable_layouts = find_suitable_layouts_for_content(
            len(image_elements), 
            len(text_elements), 
            extracted_bboxes, 
            min_text_area=10000  # Changed from 24000 to 10000
        )
        
        # Fallback mechanism if no suitable layout found
        if not suitable_layouts:
            print(f"  Warning: No suitable layout found for wiki {wiki_id}")
            print(f"    Trying fallback: reducing text area requirement to 5000 px²")
            
            # Try with lower text area threshold
            suitable_layouts = find_suitable_layouts_for_content(
                len(image_elements), 
                len(text_elements), 
                extracted_bboxes, 
                min_text_area=5000  # Reduced threshold
            )
            
            if not suitable_layouts:
                print(f"    Still no layout found, trying with relaxed constraints...")
                
                # Last resort: find layouts that can fit at least some of the content
                for bbox_data in extracted_bboxes:
                    bboxes = bbox_data['bboxes']
                    element_bboxes = [b for b in bboxes if b.get('category') == 'element']
                    text_bboxes = [b for b in bboxes if b.get('category') == 'text']
                    
                    # Accept if layout has at least 50% capacity
                    if len(element_bboxes) >= len(image_elements) * 0.5 and len(text_bboxes) >= len(text_elements) * 0.5:
                        suitable_layouts.append({
                            'layout': bbox_data,
                            'score': 0.5,  # Low score indicates fallback
                            'element_capacity': len(element_bboxes),
                            'text_capacity': len(text_bboxes)
                        })
                
                if suitable_layouts:
                    print(f"    Found {len(suitable_layouts)} fallback layouts (50% capacity)")
                    stats['fallback_used'] += 1
                else:
                    print(f"    ERROR: No fallback layout found, SKIPPING wiki {wiki_id}")
                    print(f"    This infographic will be MISSING from output!")
                    stats['skipped'] += 1
                    stats['skipped_wiki_ids'].append(wiki_id)
                    continue
        
        # Store for second pass
        layout_assignments.append({
            'wiki_idx': wiki_idx,
            'wiki_id': wiki_id,
            'infographic': infographic,
            'image_elements': image_elements,
            'text_elements': text_elements,
            'full_image_caption': full_image_caption,
            'suitable_layouts': suitable_layouts
        })
    
    # Second pass: assign layouts using round-robin with shuffling
    print("\n=== Phase 2: Assigning layouts using round-robin with shuffling ===")
    
    # Create a pool of layout indices with their capabilities
    # We'll create a shuffled round-robin queue
    layout_pool = {}  # Key: (image_count, text_count), Value: list of suitable layout indices
    
    # Group by requirements
    for assignment in layout_assignments:
        key = (len(assignment['image_elements']), len(assignment['text_elements']))
        if key not in layout_pool:
            layout_pool[key] = []
        layout_pool[key].append(assignment)
    
    # For each group, shuffle and assign layouts in round-robin fashion
    for key, assignments in layout_pool.items():
        image_count, text_count = key
        print(f"\nProcessing group: {image_count} images, {text_count} texts ({len(assignments)} infographics)")
        
        # Get all suitable layout indices from first assignment (they should be similar)
        if assignments:
            all_suitable_layout_indices = [sl['layout']['index'] for sl in assignments[0]['suitable_layouts']]
            
            # Shuffle the layout pool for randomness
            shuffled_layouts = all_suitable_layout_indices.copy()
            random.shuffle(shuffled_layouts)
            
            print(f"  Available layouts: {len(shuffled_layouts)} layouts")
            print(f"  Layout pool (shuffled): {shuffled_layouts[:20]}{'...' if len(shuffled_layouts) > 20 else ''}")
            
            # Assign layouts in round-robin fashion
            for idx, assignment in enumerate(assignments):
                # Use modulo to cycle through the shuffled layout pool
                layout_idx_in_pool = idx % len(shuffled_layouts)
                selected_layout_index = shuffled_layouts[layout_idx_in_pool]
                
                # Find the actual layout data
                selected_layout = next(
                    (sl['layout'] for sl in assignment['suitable_layouts'] 
                     if sl['layout']['index'] == selected_layout_index),
                    assignment['suitable_layouts'][0]['layout']  # Fallback to best layout
                )
                
                assignment['selected_layout'] = selected_layout
                print(f"  Wiki {assignment['wiki_id']}: assigned layout {selected_layout_index}")
    
    # Third pass: process each infographic with assigned layout
    print("\n=== Phase 3: Building infographic data ===")
    for assignment in layout_assignments:
        wiki_id = assignment['wiki_id']
        infographic = assignment['infographic']
        image_elements = assignment['image_elements']
        text_elements = assignment['text_elements']
        full_image_caption = assignment['full_image_caption']
        best_layout = assignment['selected_layout']
        
        print(f"\nProcessing wiki {wiki_id}:")
        
        selected_bbox_index = best_layout['index']
        bboxes = best_layout['bboxes']
        
        # Extract font and colors from the selected layout
        font_token, layout_color_ids = extract_font_color_from_bboxes(bboxes, font_idx)

        # Convert color IDs to color names, excluding white
        layout_colors = []
        for color_id in layout_color_ids:
            color_name = get_color_name_from_id(color_id, color_idx)
            if color_name != 'white':  # Exclude white color
                layout_colors.append(color_name)

        # If no valid colors found or all were white, use a default set without white
        if not layout_colors:
            num_colors = random.randint(1, 4)
            layout_colors = get_random_colors(color_idx, num_colors)

        # Get background bbox from the layout (category "background")
        background_bboxes = [b for b in bboxes if b.get('category') == 'background']

        # Separate full image elements and regular elements (exclude background)
        element_bboxes = [b for b in bboxes if b.get('category') == 'element']
        full_image_elements = [b for b in element_bboxes if b['bottom_right'] == [896, 2240]]
        regular_elements = [b for b in element_bboxes if b['bottom_right'] != [896, 2240]]

        # Clean caption (remove tags but keep content)
        cleaned_caption = clean_caption_text(full_image_caption)

        # ===== BUILD FULL VERSION =====
        output_layers = []

        # Layer 0: Add base layer (full image, always first)
        base_layer = {
            'category': 'base',
            'top_left': [0, 0],
            'bottom_right': [896, 2240],
            'caption': cleaned_caption
        }
        output_layers.append(base_layer)

        # Layer 1: Add background layer from extracted_bboxes (always second)
        # Background should use the background bbox from the layout
        if len(background_bboxes) > 0:
            bg_bbox = background_bboxes[0]
            bg_layer = {
                'category': 'element',
                'top_left': bg_bbox['top_left'],
                'bottom_right': bg_bbox['bottom_right'],
                'caption': bg_bbox.get('caption', '')  # Use caption from extracted background
            }
            output_layers.append(bg_layer)
        else:
            # Fallback: if no background bbox found, use a default full image background with default caption
            bg_layer = {
                'category': 'element',
                'top_left': [0, 0],
                'bottom_right': [896, 2240],
                'caption': "The image you've provided is completely blank and white. There are no objects, no text, no colors, and no discernible features. It's a simple, unadorned white background with no additional elements."
            }
            output_layers.append(bg_layer)

        # Select figure bboxes to match all image elements with better canvas coverage
        num_figures_needed = len(image_elements)
        
        # Use area-weighted spatial distribution algorithm
        selected_figure_bboxes = select_spatially_distributed_bboxes(
            regular_elements, num_figures_needed, canvas_height=2240
        )
        print(f"  Selected {len(selected_figure_bboxes)}/{num_figures_needed} figure bboxes")

        # Add image elements as 'element' category
        for idx, bbox in enumerate(selected_figure_bboxes):
            if idx < len(image_elements):
                img_elem = image_elements[idx]
                # Use the full description including "Figure: " prefix
                caption = img_elem['description']
            else:
                # Fallback caption
                is_full_image = bbox['bottom_right'] == [896, 2240]
                if is_full_image:
                    caption = "background image"
                else:
                    caption = "decorative element"

            output_layer = {
                'category': 'element',
                'top_left': bbox['top_left'],
                'bottom_right': bbox['bottom_right'],
                'caption': caption
            }
            output_layers.append(output_layer)

        # Select bboxes for text elements with title priority and smart text-to-bbox matching
        text_count = len(text_elements)
        
        if text_count > 0:
            # Get all text bboxes
            all_text_bboxes = [b for b in bboxes if b.get('category') == 'text']
            
            if text_count == 1:
                # Only title: Select the largest text bbox
                selected_text_bboxes = select_largest_non_overlapping_bboxes(bboxes, 'text', 1, min_area=0)
                text_to_bbox_mapping = [(0, selected_text_bboxes[0])] if selected_text_bboxes else []
            else:
                # Multiple texts: Smart matching based on text length
                
                # Step 1: Classify texts by word count (title is always index 0)
                title_text = text_elements[0]
                other_texts = text_elements[1:]
                
                # Classify other texts: long (>15 words) vs short (≤15 words)
                long_texts = []  # (original_index, text)
                short_texts = []  # (original_index, text)
                
                for i, text in enumerate(other_texts, start=1):  # start=1 because index 0 is title
                    word_count = len(text.split())
                    if word_count > 15:
                        long_texts.append((i, text))
                    else:
                        short_texts.append((i, text))
                
                print(f"    Text classification: 1 title, {len(long_texts)} long texts (>15 words), {len(short_texts)} short texts (≤15 words)")
                
                # Step 2: Get title bbox (largest bbox)
                title_bboxes = select_largest_non_overlapping_bboxes(bboxes, 'text', 1, min_area=0)
                
                # Step 3: Get other text candidates (area >= 10000, excluding title)
                other_candidates = [b for b in all_text_bboxes 
                                  if calculate_bbox_area(b) >= 10000]
                
                # Remove title bbox from candidates if it exists
                if title_bboxes:
                    title_bbox = title_bboxes[0]
                    other_candidates = [bbox for bbox in other_candidates 
                                      if not (bbox['top_left'] == title_bbox['top_left'] and 
                                            bbox['bottom_right'] == title_bbox['bottom_right'])]
                
                # Step 4: Assign bboxes to texts based on size matching and spatial distribution
                text_to_bbox_mapping = []
                
                # Title gets largest bbox
                if title_bboxes:
                    text_to_bbox_mapping.append((0, title_bboxes[0]))
                
                # Step 4a: Long texts get larger bboxes (sort by area, take largest ones)
                other_candidates.sort(key=calculate_bbox_area, reverse=True)
                num_long_bboxes = min(len(long_texts), len(other_candidates))
                
                # Track which bboxes have been assigned
                assigned_bboxes = set()
                if title_bboxes:
                    assigned_bboxes.add((tuple(title_bboxes[0]['top_left']), tuple(title_bboxes[0]['bottom_right'])))
                
                # Assign largest bboxes to long texts
                long_bbox_assignments = []
                for i, (text_idx, _) in enumerate(long_texts[:num_long_bboxes]):
                    bbox = other_candidates[i]
                    bbox_key = (tuple(bbox['top_left']), tuple(bbox['bottom_right']))
                    assigned_bboxes.add(bbox_key)
                    long_bbox_assignments.append((text_idx, bbox))
                    text_to_bbox_mapping.append((text_idx, bbox))
                
                # Step 4b: Short texts get spatially distributed bboxes from remaining candidates
                # Get remaining unassigned candidates
                remaining_candidates = [bbox for bbox in other_candidates 
                                       if (tuple(bbox['top_left']), tuple(bbox['bottom_right'])) not in assigned_bboxes]
                
                if short_texts and remaining_candidates:
                    # Use spatial distribution algorithm for short texts
                    num_short_needed = len(short_texts)
                    
                    print(f"    Selecting {num_short_needed} spatially distributed bboxes for short texts from {len(remaining_candidates)} candidates")
                    
                    spatially_distributed_bboxes = select_spatially_distributed_bboxes(
                        remaining_candidates,
                        num_short_needed,
                        canvas_height=2240
                    )
                    
                    # Assign spatially distributed bboxes to short texts
                    for i, (text_idx, _) in enumerate(short_texts[:len(spatially_distributed_bboxes)]):
                        text_to_bbox_mapping.append((text_idx, spatially_distributed_bboxes[i]))
                
                # Sort mapping by original text index to maintain order
                text_to_bbox_mapping.sort(key=lambda x: x[0])
                
                # Extract just the bboxes for backward compatibility with existing code
                selected_text_bboxes = [bbox for _, bbox in text_to_bbox_mapping]
                
                num_short_assigned = len([idx for idx, _ in text_to_bbox_mapping if idx > 0 and idx in [t[0] for t in short_texts]])
                print(f"    Bbox assignment: {len(long_texts)} long texts → {num_long_bboxes} large bboxes, "
                      f"{len(short_texts)} short texts → {num_short_assigned} spatially distributed bboxes")
        else:
            selected_text_bboxes = []
            text_to_bbox_mapping = []
        
        print(f"  Selected {len(selected_text_bboxes)}/{text_count} text bboxes (title: largest, long texts: larger bboxes, short texts: smaller bboxes)")

        # Add text elements as 'text' category using the smart mapping
        for text_idx, bbox in text_to_bbox_mapping:
            text_content = text_elements[text_idx]
            
            # Chọn màu từ layout_colors (không có thì random, đã loại màu trắng)
            color_name = random.choice(layout_colors) if layout_colors else 'black'
            # Replace white/near-white colors with black for readability
            color_name = replace_white_color_with_black(color_name, color_idx)
            color_id = color_idx[color_name]

            # Dùng font_token cho toàn bộ layout
            caption = f'Text "{text_content}" in <color-{color_id}>, <{font_token}>. '

            output_layer = {
                'category': 'text',
                'top_left': bbox['top_left'],
                'bottom_right': bbox['bottom_right'],
                'caption': caption,
                'text': text_content
            }
            output_layers.append(output_layer)
        
        # Report warnings for texts without bboxes
        assigned_indices = {text_idx for text_idx, _ in text_to_bbox_mapping}
        for idx in range(text_count):
            if idx not in assigned_indices:
                print(f"    Warning: No bbox available for text element {idx+1}: '{text_elements[idx][:50]}...'")
        
        # Report final counts
        final_elements = len([l for l in output_layers if l['category'] == 'element' and 'background' not in l.get('caption', '').lower()])
        final_texts = len([l for l in output_layers if l['category'] == 'text'])
        print(f"  Final output: {final_elements} figure elements, {final_texts} text elements")
        
        # Extract context and QA from infographic data
        context = infographic.get('context', '')
        qa_ids = infographic.get('ids', [])
        
        # Map IDs to original QA pairs from SQuAD dataset
        original_qa_pairs = []
        for qa_id in qa_ids:
            if qa_id in squad_id_to_qa:
                original_qa_pairs.append(squad_id_to_qa[qa_id])
            else:
                print(f"  Warning: QA ID {qa_id} not found in SQuAD dataset")
        
        # Create the final structure with unique wiki ID
        result_item = {
            'index': wiki_id,
            'layers_all': output_layers,
            'full_image_caption': full_image_caption,  # Keep original caption without cleaning
            'original_bbox_index': selected_bbox_index,
            'context': context,
            'original_qa_pairs': original_qa_pairs
        }
        result.append(result_item)
        stats['successful'] += 1
    
    # Print statistics summary
    print("\n" + "="*60)
    print("PROCESSING STATISTICS")
    print("="*60)
    print(f"Total input infographics: {stats['total_infographics']}")
    print(f"Successfully processed: {stats['successful']} ({stats['successful']/stats['total_infographics']*100:.1f}%)")
    print(f"Fallback layouts used: {stats['fallback_used']} ({stats['fallback_used']/stats['total_infographics']*100:.1f}%)")
    print(f"Skipped (no layout): {stats['skipped']} ({stats['skipped']/stats['total_infographics']*100:.1f}%)")
    
    if stats['skipped'] > 0:
        print(f"\nWARNING: {stats['skipped']} infographics were SKIPPED!")
        print(f"Skipped wiki IDs: {stats['skipped_wiki_ids'][:10]}")
        if len(stats['skipped_wiki_ids']) > 10:
            print(f"... and {len(stats['skipped_wiki_ids']) - 10} more")
        print("\nThese infographics will be MISSING from the output files.")
        print("Consider:")
        print("  1. Using layouts with more flexibility")
        print("  2. Reducing text/image requirements")
        print("  3. Adding more diverse layouts to extracted_bboxes.json")
    
    print("="*60 + "\n")
    
    return result


def update_original_wiki_files(merged_data: List[Dict], original_wiki_dir: str):
    """
    Update original wiki files by replacing entries that match the indices in merged_data.
    
    Args:
        merged_data: List of merged infographic data with 'index' field
        original_wiki_dir: Path to directory containing original wiki*.json files
    """
    if not os.path.exists(original_wiki_dir):
        print(f"\nWarning: Original wiki directory does not exist: {original_wiki_dir}")
        print("Skipping original file updates.")
        return
    
    print("\n" + "="*60)
    print("UPDATING ORIGINAL WIKI FILES")
    print("="*60)
    print(f"Original wiki directory: {original_wiki_dir}")
    
    # Group updates by file
    updates_by_file = {}  # file_index -> list of (position_in_array, new_entry)
    
    for entry in merged_data:
        wiki_id = entry['index']
        
        # Calculate which file this entry belongs to
        # wiki_id 1-50 → file 1, wiki_id 51-100 → file 2, etc.
        file_index = ((wiki_id - 1) // 50) + 1
        position_in_array = (wiki_id - 1) % 50
        
        if file_index not in updates_by_file:
            updates_by_file[file_index] = []
        
        updates_by_file[file_index].append((position_in_array, entry))
    
    print(f"Total entries to update: {len(merged_data)}")
    print(f"Files to be modified: {len(updates_by_file)}")
    
    # Update each file
    updated_files = 0
    updated_entries = 0
    
    for file_index in sorted(updates_by_file.keys()):
        filename = f"wiki{file_index:06d}.json"
        filepath = os.path.join(original_wiki_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"\nWarning: Original file not found: {filename}")
            print(f"   Skipping updates for file index {file_index}")
            continue
        
        try:
            # Load original file
            original_data = load_json(filepath)
            
            if not isinstance(original_data, list):
                print(f"\nError: {filename} is not a JSON array, skipping")
                continue
            
            # Apply updates
            entries_updated_in_file = 0
            updates_for_file = updates_by_file[file_index]
            
            for position, new_entry in updates_for_file:
                if position >= len(original_data):
                    print(f"   Warning: Position {position} out of range in {filename} (size: {len(original_data)})")
                    continue
                
                # Verify the index matches
                old_entry = original_data[position]
                old_index = old_entry.get('index')
                new_index = new_entry['index']
                
                if old_index != new_index:
                    print(f"   Warning: Index mismatch at position {position} in {filename}")
                    print(f"            Expected index {new_index}, found {old_index}")
                    continue
                
                # Replace the entry
                original_data[position] = new_entry
                entries_updated_in_file += 1
                updated_entries += 1
            
            # Save updated file
            save_json(original_data, filepath)
            updated_files += 1
            
            # Get list of updated indices for this file
            updated_indices = [entry['index'] for _, entry in updates_for_file]
            print(f"\nUpdated {filename}: {entries_updated_in_file} entries")
            print(f"  Wiki IDs: {updated_indices[:5]}{'...' if len(updated_indices) > 5 else ''}")
            
        except Exception as e:
            print(f"\nError updating {filename}: {e}")
            continue
    
    print("\n" + "="*60)
    print("UPDATE SUMMARY")
    print("="*60)
    print(f"Files modified: {updated_files}/{len(updates_by_file)}")
    print(f"Entries updated: {updated_entries}/{len(merged_data)}")
    print("="*60 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Merge narrator-generated infographic data with extracted bounding boxes'
    )
    parser.add_argument(
        '--extracted-bboxes',
        type=str,
        default="src/data/narrator/extracted_bboxes.json",
        help='Path to extracted_bboxes.json'
    )
    parser.add_argument(
        '--infographic-dir',
        type=str,
        default="./src/data/create_data/output/infographic",
        help='Directory containing infographic*.json files (default: src/data/create_data/output/infographic). Ignored if --infor_path is specified.'
    )
    parser.add_argument(
        '--infor_path',
        type=str,
        default=None,
        help='Path to a single infographic JSON file (e.g., failed.json). If specified, --infographic-dir, --start, and --end are ignored.'
    )
    parser.add_argument(
        '--color-idx',
        type=str,
        default="src/data/narrator/glyph/color_idx.json",
        help='Path to color_idx.json'
    )
    parser.add_argument(
        '--font-idx',
        type=str,
        default="src/data/narrator/glyph/font_idx.json",
        help='Path to font_idx.json'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='Start file index for infographic generation (inclusive, 1-based). Ignored if --infor_path is specified.'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End file index for infographic generation (exclusive, 1-based). Ignored if --infor_path is specified.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: src/data/create_data/output/narrator_format)'
    )
    parser.add_argument(
        '--squad-file',
        type=str,
        default="/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl",
        help='Path to squad_v2_train.jsonl file (default: /mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl)'
    )
    parser.add_argument(
        '--original-wiki-dir',
        type=str,
        default="./src/data/narrator/wiki",
        help='Path to original wiki directory for updating entries when using --infor_path (default: ./src/data/narrator/wiki)'
    )

    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths
    extracted_bboxes_path = args.extracted_bboxes
    color_idx_path = args.color_idx
    font_idx_path = args.font_idx
    
    # Determine infographic data source
    if args.infographic_dir:
        infographic_dir = args.infographic_dir
    else:
        # Default to output/infographic directory
        repo_root = os.path.abspath(os.path.join(script_dir, '../../../'))
        infographic_dir = os.path.join(repo_root, 'src/data/create_data/output/infographic')
    
    # Check if we're using single file mode
    use_single_file = args.infor_path is not None
    
    # Set base output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to src/data/create_data/output/narrator_format from repository root
        repo_root = os.path.abspath(os.path.join(script_dir, '../../../'))
        output_dir = os.path.join(repo_root, 'src/data/create_data/output/narrator_format')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    print(f"  - Bboxes: {extracted_bboxes_path}")
    print(f"  - Colors: {color_idx_path}")
    print(f"  - Fonts: {font_idx_path}")
    print(f"  - SQuAD: {args.squad_file}")
    
    extracted_bboxes = load_json(extracted_bboxes_path)
    color_idx = load_json(color_idx_path)
    font_idx = load_json(font_idx_path)
    squad_id_to_qa = load_squad_jsonl(args.squad_file)
    
    # Load infographic data based on mode
    if use_single_file:
        # Single file mode: load from --infor_path
        print(f"  - Infographics: {args.infor_path} (single file)")
        
        if not os.path.exists(args.infor_path):
            raise FileNotFoundError(f"Infographic file not found: {args.infor_path}")
        
        infographic_generated = load_json(args.infor_path)
        
        # Ensure it's a list
        if not isinstance(infographic_generated, list):
            raise ValueError(f"Expected a list of infographic entries, got {type(infographic_generated)}")
        
        print(f"Loaded {len(infographic_generated)} infographic entries from single file")
        
        # In single file mode, use the infographic_id from each entry directly
        start_wiki_idx = 0
    else:
        # Directory mode: load from --infographic-dir with --start and --end
        end_file_idx = args.end if args.end is not None else (args.start + 100)  # Default to 100 files
        
        # Validate file indices
        if args.start < 1:
            raise ValueError("Start file index must be >= 1")
        if args.end is not None and args.end <= args.start:
            raise ValueError("End file index must be > start file index")
        
        print(f"  - Infographics: {infographic_dir} (directory)")
        print(f"  - File index range: {args.start} to {end_file_idx-1} (files: {end_file_idx - args.start})")
        
        infographic_generated = load_infographic_files_from_directory(
            infographic_dir, 
            args.start, 
            end_file_idx
        )
        
        # Calculate starting wiki index based on start file index
        start_wiki_idx = (args.start - 1) * 50
    
    print(f"\nLoaded {len(extracted_bboxes)} bbox entries")
    print(f"Loaded {len(color_idx)} colors")
    print(f"Loaded {len(font_idx)} fonts")
    print(f"Loaded {len(squad_id_to_qa)} SQuAD QA pairs")
    
    # Process data
    print("\nProcessing data...")
    
    merged_data = merge_narrator_data(
        infographic_generated,
        extracted_bboxes,
        color_idx,
        font_idx,
        squad_id_to_qa,
        start_wiki_idx=start_wiki_idx,
        use_infographic_id=use_single_file  # Use infographic_id when in single file mode
    )
    
    print(f"Generated {len(merged_data)} infographics")
    
    # Save results in chunks of 50 per file
    chunk_size = 50
    
    # Save results
    print(f"\nSaving to {output_dir}...")
    saved_files = []
    
    for i in range(0, len(merged_data), chunk_size):
        chunk = merged_data[i:i + chunk_size]
        
        # Calculate file index based on position in merged_data and start file index
        file_index = args.start + (i // chunk_size)
        
        filename = f"wiki{file_index:06d}.json"
        filepath = os.path.join(output_dir, filename)
        
        save_json(chunk, filepath)
        saved_files.append(filename)
        
        print(f"  Saved {len(chunk)} infographics to {filename} (wiki IDs: {chunk[0]['index']}-{chunk[-1]['index']})")
    
    print(f"\nSaved {len(saved_files)} files total")
    
    # Update original wiki files if in single file mode (--infor_path)
    if use_single_file:
        print("\n" + "="*60)
        print("SINGLE FILE MODE DETECTED")
        print("="*60)
        print(f"Output saved to: {output_dir}")
        print(f"Now updating original wiki files...")
        
        update_original_wiki_files(merged_data, args.original_wiki_dir)
    
    print("Done!")
    
    # Print summary statistics
    print("\n=== Summary ===")
    total_layers = sum(len(item['layers_all']) for item in merged_data)
    
    if merged_data:
        first_wiki_id = merged_data[0]['index']
        last_wiki_id = merged_data[-1]['index']
        print(f"Wiki ID range: {first_wiki_id:06d} - {last_wiki_id:06d}")
    
    print(f"Total infographics: {len(merged_data)}")
    print(f"Total files saved: {len(saved_files)}")
    print(f"Total layers: {total_layers}")
    print(f"Output directory: {output_dir}")
    
    if use_single_file:
        print(f"Original wiki directory updated: {args.original_wiki_dir}")


if __name__ == '__main__':
    main()