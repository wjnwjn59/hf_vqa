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


def extract_images_from_caption(full_caption: str) -> List[str]:
    """
    Extract image descriptions from caption using new format.
    Format: Figure: description.
    
    Args:
        full_caption: The full image caption text
        
    Returns:
        List of image descriptions (without "Figure: " prefix)
    """
    image_elements = []
    
    # Pattern to match "Figure: " followed by description up to a period
    pattern = r'Figure:\s+([^.]+\.)'
    matches = re.findall(pattern, full_caption, re.IGNORECASE)
    
    for description in matches:
        # Remove "Figure: " prefix, keep only the description
        image_elements.append({
            'description': description.strip()
        })
    
    return image_elements


def extract_images_from_figures(figures: List[Dict]) -> List[Dict]:
    """
    Extract image descriptions from figures data by randomly selecting from ideas.
    
    Args:
        figures: List of figure data with 'ideas' field
        
    Returns:
        List of image descriptions
    """
    image_elements = []
    
    for figure in figures:
        ideas = figure.get('ideas', [])
        if ideas:
            # Randomly select one idea from the available ideas
            selected_idea = random.choice(ideas)
            image_elements.append({
                'description': selected_idea.strip()
            })
        else:
            # Fallback if no ideas available
            image_elements.append({
                'description': "A simple abstract illustration relevant to the content."
            })
    
    return image_elements


def extract_text_elements(full_caption: str) -> List[str]:
    """
    Extract text content from caption (quoted text).
    Format: "text content"
    
    Args:
        full_caption: The full image caption text
        
    Returns:
        List of text strings
    """
    text_elements = []
    
    # Pattern to match quoted text
    pattern = r'"([^"]+)"'
    matches = re.findall(pattern, full_caption)
    
    for text_content in matches:
        text_elements.append(text_content.strip())
    
    return text_elements


def clean_caption_text(caption: str) -> str:
    """
    Clean up caption text - remove "Figure: " prefix and normalize spacing.
    """
    # Remove "Figure: " prefix and the description up to period
    caption = re.sub(r'Figure:\s+[^.]+\.', '', caption, flags=re.IGNORECASE)
    
    # Clean up extra whitespace and normalize spacing
    caption = re.sub(r'\s+', ' ', caption).strip()
    
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


def sort_by_reading_order(bboxes: List[Dict]) -> List[Dict]:
    """
    Sort bboxes by reading order (left to right, top to bottom).
    
    Args:
        bboxes: List of bboxes with 'top_left' coordinates
        
    Returns:
        Sorted list of bboxes
    """
    def reading_order_key(bbox):
        x, y = bbox['top_left']
        # Primary sort by y (top to bottom), secondary sort by x (left to right)
        return (y, x)
    
    return sorted(bboxes, key=reading_order_key)

def merge_narrator_data(
    infographic_generated: List[Dict],
    extracted_bboxes: List[Dict],
    color_idx: Dict,
    font_idx: Dict,
    start_wiki_idx: int = 0
) -> List[Dict]:
    """
    Process narrator-generated infographic data and merge with bboxes.
    
    Args:
        infographic_generated: List of infographic data with full_image_caption and background_caption
        extracted_bboxes: List of bbox data from extracted_bboxes.json
        color_idx: Color index mapping
        font_idx: Font index mapping
        start_wiki_idx: Starting wiki data index (0-based) for generating unique IDs
    
    Returns:
        List of merged infographic data (full layout version only)
    """
    result = []
    
    # Create a mapping of indices from extracted_bboxes
    bbox_by_index = {item['index']: item for item in extracted_bboxes}
    available_indices = list(bbox_by_index.keys())
    
    for wiki_idx, infographic in enumerate(infographic_generated):
        # Generate unique wiki index based on start_wiki_idx
        wiki_id = start_wiki_idx + wiki_idx + 1  # Start from 1, not 0
        
        # Get the caption data from generated_infographic structure
        generated_infographic = infographic.get('generated_infographic', {})
        
        # Handle case where generated_infographic is None, a JSON string, or empty
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
        
        # Extract image elements from figures.ideas (new approach)
        image_elements = extract_images_from_figures(figures)
        # Extract text elements from full_image_caption (keep original approach for "titled" content)
        text_elements = extract_text_elements(full_image_caption)
        
        # Select a random bbox index from available indices
        if not available_indices:
            print("Warning: No more bbox indices available, wrapping around")
            available_indices = list(bbox_by_index.keys())
        
        selected_bbox_index = random.choice(available_indices)
        # Remove the selected index to avoid reuse
        available_indices.remove(selected_bbox_index)
        
        bbox_data = bbox_by_index[selected_bbox_index]
        bboxes = bbox_data['bboxes']
        
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

        # Select bbox elements based on available figure descriptions
        # If not enough bbox elements, truncate figure list to match available bboxes
        num_available_elements = len(regular_elements)
        num_figures_to_use = min(len(image_elements), num_available_elements)
        
        # Truncate image_elements if needed (keep original order)
        image_elements_to_use = image_elements[:num_figures_to_use]

        # Sort regular elements by area and select largest non-overlapping
        regular_elements.sort(key=calculate_bbox_area, reverse=True)
        selected_decorative = []

        for bbox in regular_elements:
            # Only check overlap with other decorative elements
            overlaps = any(bboxes_overlap(bbox, s) for s in selected_decorative)
            if not overlaps:
                selected_decorative.append(bbox)
                if len(selected_decorative) >= num_figures_to_use:
                    break

        # Create pairs of (bbox, caption) - bbox already selected by largest area
        # Assign captions in original order to the largest bboxes
        bbox_caption_pairs = []
        for idx, bbox in enumerate(selected_decorative):
            if idx < len(image_elements_to_use):
                caption = image_elements_to_use[idx]['description']
            else:
                caption = "decorative element"  # Fallback (should not happen)
            bbox_caption_pairs.append((bbox, caption))

        # Sort pairs by reading order of bboxes (preserving caption assignment)
        bbox_caption_pairs.sort(key=lambda pair: (pair[0]['top_left'][1], pair[0]['top_left'][0]))

        # Add image elements as 'element' category with reading order of bboxes
        for bbox, caption in bbox_caption_pairs:
            output_layer = {
                'category': 'element',
                'top_left': bbox['top_left'],
                'bottom_right': bbox['bottom_right'],
                'caption': caption
            }
            output_layers.append(output_layer)

        # Select bboxes for text (non-overlapping)
        # Chỉ chọn các bbox text không phải full image
        valid_text_bboxes = [b for b in bboxes if b.get('category') == 'text' and b.get('bottom_right') != [896, 2240]]
        # Chọn không-overlap lớn nhất
        selected_text_bboxes = select_largest_non_overlapping_bboxes(valid_text_bboxes, 'text', len(text_elements))

        # Create pairs of (bbox, text_content) - bbox already selected by largest area
        # Assign text content in original order to the largest bboxes
        text_bbox_pairs = []
        for idx, bbox in enumerate(selected_text_bboxes):
            text_content = text_elements[idx] if idx < len(text_elements) else ""
            text_bbox_pairs.append((bbox, text_content))

        # Sort pairs by reading order of bboxes (preserving text assignment)
        text_bbox_pairs.sort(key=lambda pair: (pair[0]['top_left'][1], pair[0]['top_left'][0]))

        # Process text pairs (bbox selected by area, text assigned in order, sorted by reading order)
        for bbox, text_content in text_bbox_pairs:
            
            # Extract font and color from individual bbox
            bbox_font_color_info = bbox.get('font_color_info', '')
            
            # Get font from this specific bbox, fallback to layout font
            bbox_font_token = font_token  # Default fallback
            if bbox_font_color_info:
                font_match = re.search(r'<(en-font-\d+)>', bbox_font_color_info)
                if font_match:
                    bbox_font_token = font_match.group(1)
                else:
                    # If not English font, use random English font
                    en_fonts = [k for k in font_idx.keys() if k.startswith('en-')]
                    if en_fonts:
                        selected_font = random.choice(en_fonts)
                        font_id = font_idx[selected_font]
                        bbox_font_token = f'en-font-{font_id}'
            
            # Get color from this specific bbox, fallback to layout colors
            color_name = 'black'  # Default fallback
            if bbox_font_color_info:
                color_match = re.search(r'<color-(\d+)>', bbox_font_color_info)
                if color_match:
                    color_id = color_match.group(1)
                    color_name = get_color_name_from_id(int(color_id), color_idx)
                    if color_name == 'white':
                        # If white, use random from layout colors
                        color_name = random.choice(layout_colors) if layout_colors else 'black'
                else:
                    # Fallback to layout colors
                    color_name = random.choice(layout_colors) if layout_colors else 'black'
            else:
                # Fallback to layout colors
                color_name = random.choice(layout_colors) if layout_colors else 'black'
            
            color_id = color_idx[color_name]
            caption = f'Text "{text_content}" in <color-{color_id}>, <{bbox_font_token}>. '
            output_layer = {
                'category': 'text',
                'top_left': bbox['top_left'],
                'bottom_right': bbox['bottom_right'],
                'caption': caption,
                'text': text_content
            }
            output_layers.append(output_layer)
        
        # Create the final structure with unique wiki ID
        result_item = {
            'index': wiki_id,
            'layers_all': output_layers,
            'full_image_caption': full_image_caption,
            'original_bbox_index': selected_bbox_index
        }
        result.append(result_item)
    
    return result


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Merge narrator-generated infographic data with extracted bounding boxes'
    )
    parser.add_argument(
        '--extracted-bboxes',
        type=str,
        default=None,
        help='Path to extracted_bboxes.json (default: ../create_data/qwen/extracted_bboxes.json)'
    )
    parser.add_argument(
        '--infographic-dir',
        type=str,
        default="src/data/create_data/output/infographic_v2",
        help='Directory containing infographic*.json files (default: src/data/create_data/output/infographic_v2)'
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
        help='Path to font_idx.json (default: ../bizgen/glyph/font_uni_10-lang_idx.json)'
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
        help='Start file index for infographic generation (inclusive, 1-based)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End file index for infographic generation (exclusive, 1-based)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: src/data/create_data/output/narrator_format_v2)'
    )

    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths
    extracted_bboxes_path = args.extracted_bboxes or os.path.join(script_dir, '../create_data/qwen/extracted_bboxes.json')
    color_idx_path = args.color_idx or os.path.join(script_dir, '../bizgen/glyph/color_idx.json')
    font_idx_path = args.font_idx or os.path.join(script_dir, '../bizgen/glyph/font_uni_10-lang_idx.json')
    
    # Determine infographic data source
    if args.infographic_dir:
        infographic_dir = args.infographic_dir
    else:
        # Default to output/infographic_v2 directory
        repo_root = os.path.abspath(os.path.join(script_dir, '../../../'))
        infographic_dir = os.path.join(repo_root, 'src/data/create_data/output/infographic_v2')
    
    # Set base output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to src/data/create_data/output/narrator_format_v2 from repository root
        repo_root = os.path.abspath(os.path.join(script_dir, '../../../'))
        output_dir = os.path.join(repo_root, 'src/data/create_data/output/narrator_format_v2')
    
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
    
    print(f"\nLoaded {len(extracted_bboxes)} bbox entries")
    print(f"Loaded {len(color_idx)} colors")
    print(f"Loaded {len(font_idx)} fonts")
    
    # Process data
    print("\nProcessing data...")
    # Calculate starting wiki index based on start file index
    start_wiki_idx = (args.start - 1) * 50
    
    merged_data = merge_narrator_data(
        infographic_generated,
        extracted_bboxes,
        color_idx,
        font_idx,
        start_wiki_idx=start_wiki_idx
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


if __name__ == '__main__':
    main()
