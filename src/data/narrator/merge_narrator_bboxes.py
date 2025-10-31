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
        print(f"  Filtered text bboxes by area >= {min_area}: {len(filtered)} remaining")
    
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


def find_best_layout_for_content(
    image_count: int,
    text_count: int,
    extracted_bboxes: List[Dict],
    min_text_area: int = 36000
) -> Dict:
    """
    Find the best layout that can accommodate the required number of images and text elements.
    Uses greedy algorithm to score each layout based on capacity.
    
    Args:
        image_count: Number of image elements needed
        text_count: Number of text elements needed
        extracted_bboxes: List of all available layouts
        min_text_area: Minimum area for text bboxes
    
    Returns:
        Best layout dict or None if no suitable layout found
    """
    best_score = -1
    best_layout = None
    
    print(f"  Searching for layout: need {image_count} images, {text_count} texts (min area: {min_text_area})")
    
    for bbox_data in extracted_bboxes:
        bboxes = bbox_data['bboxes']
        
        # Count available element bboxes (excluding background)
        element_bboxes = [b for b in bboxes if b.get('category') == 'element']
        regular_elements = [b for b in element_bboxes if b['bottom_right'] != [896, 2240]]
        
        # Count available text bboxes with sufficient area
        text_bboxes = [b for b in bboxes if b.get('category') == 'text']
        valid_text_bboxes = [b for b in text_bboxes if calculate_bbox_area(b) >= min_text_area]
        
        # Apply non-overlapping selection to get realistic counts
        available_elements = select_largest_non_overlapping_bboxes(bboxes, 'element', image_count + 10)  # Extra margin
        available_texts = select_largest_non_overlapping_bboxes(bboxes, 'text', text_count + 5, min_text_area)  # Extra margin
        
        element_capacity = len(available_elements)
        text_capacity = len(available_texts)
        
        # Calculate score based on how well this layout matches requirements
        element_score = min(element_capacity, image_count) / max(image_count, 1)
        text_score = min(text_capacity, text_count) / max(text_count, 1)
        
        # Bonus for exact match or having more capacity than needed
        element_bonus = 1.0 if element_capacity >= image_count else 0.0
        text_bonus = 1.0 if text_capacity >= text_count else 0.0
        
        # Combined score (weighted towards text since it has area constraints)
        total_score = (element_score * 0.4 + text_score * 0.6) + (element_bonus * 0.2 + text_bonus * 0.3)
        
        if total_score > best_score:
            best_score = total_score
            best_layout = bbox_data
            
        print(f"    Layout {bbox_data['index']}: elements={element_capacity}/{image_count}, texts={text_capacity}/{text_count}, score={total_score:.3f}")
    
    if best_layout:
        print(f"  Selected layout {best_layout['index']} with score {best_score:.3f}")
    else:
        print(f"  Warning: No suitable layout found!")
    
    return best_layout


def merge_narrator_data(
    infographic_generated: List[Dict],
    extracted_bboxes: List[Dict],
    color_idx: Dict,
    font_idx: Dict,
    start_wiki_idx: int = 0
) -> List[Dict]:
    """
    Process narrator-generated infographic data and merge with bboxes.
    Uses greedy algorithm to find best layout for each infographic.
    
    Args:
        infographic_generated: List of infographic data with full_image_caption and background_caption
        extracted_bboxes: List of bbox data from extracted_bboxes.json
        color_idx: Color index mapping
        font_idx: Font index mapping
        start_wiki_idx: Starting wiki index for generating unique IDs
    
    Returns:
        List of merged infographic data (full layout version only)
    """
    result = []
    
    for wiki_idx, infographic in enumerate(infographic_generated):
        # Generate unique wiki index based on start_wiki_idx
        wiki_id = start_wiki_idx + wiki_idx + 1  # Start from 1, not 0
        
        # Get the caption data from generated_infographic structure
        generated_infographic = infographic.get('generated_infographic', {})
        
        # Handle case where generated_infographic is a JSON string instead of dict
        if isinstance(generated_infographic, str):
            try:
                generated_infographic = json.loads(generated_infographic)
            except:
                print(f"Warning: Could not parse generated_infographic for wiki {wiki_id}")
                continue
        
        full_image_caption = generated_infographic.get('full_image_caption', '')
        background_caption = generated_infographic.get('background_caption', '')
        
        if not full_image_caption:
            print(f"Warning: No full_image_caption for wiki {wiki_id}, skipping")
            continue
        
        # Extract image and text elements
        image_elements = extract_images_from_caption(full_image_caption)
        text_elements = extract_text_elements(full_image_caption)
        
        print(f"\nProcessing wiki {wiki_id}:")
        print(f"  Found {len(image_elements)} image elements, {len(text_elements)} text elements")
        
        # Find best layout using greedy algorithm
        best_layout = find_best_layout_for_content(
            len(image_elements), 
            len(text_elements), 
            extracted_bboxes, 
            min_text_area=36000
        )
        
        if not best_layout:
            print(f"  Warning: No suitable layout found for wiki {wiki_id}, skipping")
            continue
            
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

        # Select figure bboxes to match all image elements
        num_figures_needed = len(image_elements)
        
        # Sort regular elements by area and select largest non-overlapping
        regular_elements.sort(key=calculate_bbox_area, reverse=True)
        selected_decorative = []

        for bbox in regular_elements:
            # Only check overlap with other decorative elements
            overlaps = any(bboxes_overlap(bbox, s) for s in selected_decorative)
            if not overlaps:
                selected_decorative.append(bbox)
                if len(selected_decorative) >= num_figures_needed:
                    break

        # Use selected decorative elements for figures
        selected_figure_bboxes = selected_decorative[:num_figures_needed]
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

        # Select bboxes for text to match all text elements (non-overlapping, area >= 36000)
        text_count = len(text_elements)
        selected_text_bboxes = select_largest_non_overlapping_bboxes(bboxes, 'text', text_count, min_area=36000)
        
        print(f"  Selected {len(selected_text_bboxes)}/{text_count} text bboxes")

        # Add text elements as 'text' category
        # Special handling: First text (title) gets the largest bbox, others follow bbox size order
        for idx, text_content in enumerate(text_elements):
            if idx < len(selected_text_bboxes):
                # For title (first text), always use the largest bbox (index 0)
                # For other texts, use remaining bboxes in order
                if idx == 0:
                    bbox = selected_text_bboxes[0]  # Largest bbox for title
                else:
                    bbox = selected_text_bboxes[idx]  # Regular assignment for others
                
                # Chọn màu từ layout_colors (không có thì random, đã loại màu trắng)
                color_name = random.choice(layout_colors) if layout_colors else 'black'
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
            else:
                print(f"    Warning: No bbox available for text element {idx+1}: '{text_content[:50]}...'")
        
        # Report final counts
        final_elements = len([l for l in output_layers if l['category'] == 'element' and 'background' not in l.get('caption', '').lower()])
        final_texts = len([l for l in output_layers if l['category'] == 'text'])
        print(f"  Final output: {final_elements} figure elements, {final_texts} text elements")
        
        # Extract context and QA from infographic data
        context = infographic.get('context', '')
        qa_pairs = infographic.get('qa', [])
        
        # Create the final structure with unique wiki ID
        result_item = {
            'index': wiki_id,
            'layers_all': output_layers,
            'full_image_caption': full_image_caption,  # Keep original caption without cleaning
            'original_bbox_index': selected_bbox_index,
            'context': context,
            'qa': qa_pairs
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
        default="./src/data/narrator/extracted_bboxes.json",
        help='Path to extracted_bboxes.json'
    )
    parser.add_argument(
        '--infographic-dir',
        type=str,
        default="src/data/create_data/output/infographic",
        help='Directory containing infographic*.json files (default: src/data/create_data/output/infographic)'
    )
    parser.add_argument(
        '--color-idx',
        type=str,
        default="./src/data/narrator/glyph/color_idx.json",
        help='Path to color_idx.json'
    )
    parser.add_argument(
        '--font-idx',
        type=str,
        default="./src/data/narrator/glyph/font_idx.json",
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
        help='Output directory (default: src/data/create_data/output/narrator_format)'
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