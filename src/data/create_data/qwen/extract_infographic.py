import json
import random
import re
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
    Each file contains 50 entries with infographic_id starting from (file_index-1)*50 + 1
    
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
    
    # Calculate which files to load based on wiki ID range
    # Each file contains 50 entries, file index is 1-based
    start_file_index = (start_wiki // 50) + 1
    end_file_index = ((end_wiki - 1) // 50) + 1
    
    print(f"Loading files infographic{start_file_index:06d}.json to infographic{end_file_index:06d}.json")
    
    for file_index in range(start_file_index, end_file_index + 1):
        filename = f"infographic{file_index:06d}.json"
        filepath = os.path.join(directory_path, filename)
        
        if not os.path.exists(filepath):
            print(f"  Warning: File {filename} does not exist")
            continue
            
        try:
            file_data = load_json(filepath)
            print(f"  Loaded {len(file_data)} entries from {filename}")
            
            # Filter entries based on infographic_id within range
            for entry in file_data:
                if 'infographic_id' in entry:
                    infographic_id = entry['infographic_id']
                    if start_wiki < infographic_id <= end_wiki:  # Note: start_wiki < id <= end_wiki
                        infographic_data.append(entry)
            
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue
    
    print(f"Total loaded infographic entries in range [{start_wiki+1}:{end_wiki}]: {len(infographic_data)}")
    return infographic_data


def save_json(data: Any, filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


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


def parse_coordinates(coord_str: str) -> List[int]:
    """
    Parse coordinate string into bounding box coordinates.
    Expected format: [x_min, y_min, x_max, y_max] (4 integers)
    """
    try:
        # Remove whitespace and parse as JSON
        coord_str = coord_str.strip()
        coords = json.loads(coord_str)
        # Validate that we have exactly 4 integer coordinates
        if len(coords) == 4 and all(isinstance(coord, int) for coord in coords):
            return coords
        return None
    except:
        return None


def extract_text_with_coordinates(caption: str) -> List[Dict]:
    """
    Extract text content with coordinates from caption.
    Format: "text content" [x_min, y_min, x_max, y_max]
    """
    text_elements = []
    
    # Pattern to match "text" followed by coordinates [x_min, y_min, x_max, y_max]
    pattern = r'"([^"]*)".*?(\[[^\]]+\])'
    matches = re.findall(pattern, caption)
    
    for text_content, coord_str in matches:
        coordinates = parse_coordinates(coord_str)
        if coordinates and len(coordinates) == 4:
            # Format: [x_min, y_min, x_max, y_max]
            top_left = [coordinates[0], coordinates[1]]  # [x_min, y_min]
            bottom_right = [coordinates[2], coordinates[3]]  # [x_max, y_max]
            
            text_elements.append({
                'text': text_content.strip(),
                'top_left': top_left,
                'bottom_right': bottom_right
            })
    
    return text_elements


def extract_images_with_coordinates(caption: str) -> List[Dict]:
    """
    Extract image descriptions with coordinates from caption.
    Format: <figure> [class] description </figure> [x_min, y_min, x_max, y_max]
    """
    image_elements = []
    
    # Pattern to match <figure> [class] content </figure> followed by coordinates [x_min, y_min, x_max, y_max]
    pattern = r'<figure>\s*\[([^\]]*)\]\s*([^<]*)</figure>.*?(\[[^\]]+\])'
    matches = re.findall(pattern, caption)
    
    for figure_class, image_desc, coord_str in matches:
        coordinates = parse_coordinates(coord_str)
        if coordinates and len(coordinates) == 4:
            # Format: [x_min, y_min, x_max, y_max]
            top_left = [coordinates[0], coordinates[1]]  # [x_min, y_min]
            bottom_right = [coordinates[2], coordinates[3]]  # [x_max, y_max]
            
            image_elements.append({
                'description': image_desc.strip(),
                'figure_class': figure_class.strip(),  # Store but don't use in layout
                'top_left': top_left,
                'bottom_right': bottom_right
            })
    
    return image_elements


def clean_caption_text(caption: str) -> str:
    """
    Remove coordinates and figure tags from caption, but keep text content.
    """
    # Remove coordinate patterns [...] including any whitespace before them
    caption = re.sub(r'\s*\[[^\]]+\]', '', caption)
    
    # Remove <figure> tags with class but keep content inside
    # Pattern: <figure> [class] content </figure> -> content
    caption = re.sub(r'<figure>\s*\[[^\]]*\]\s*([^<]*)</figure>', r'\1', caption)
    
    # Remove any remaining <figure> or </figure> tags
    caption = re.sub(r'</?figure>', '', caption)
    
    # Also handle old <image> format if any remain
    caption = re.sub(r'<image>([^<]*)</image>', r'\1', caption)
    
    # Clean up extra whitespace and normalize spacing
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    return caption


def is_valid_bbox(top_left: List[int], bottom_right: List[int], max_width: int = 896, max_height: int = 2240) -> bool:
    """
    Validate that bounding box is within the canvas dimensions.
    Canvas size is 896 (width) × 2240 (height).
    
    Args:
        top_left: [x_min, y_min]
        bottom_right: [x_max, y_max]
        max_width: Maximum width (default: 896)
        max_height: Maximum height (default: 2240)
    
    Returns:
        True if bbox is valid, False otherwise
    """
    if len(top_left) != 2 or len(bottom_right) != 2:
        return False
    
    x_min, y_min = top_left
    x_max, y_max = bottom_right
    
    # Check if coordinates are within bounds
    if x_min < 0 or y_min < 0:
        return False
    if x_max > max_width or y_max > max_height:
        return False
    
    # Check if bbox makes sense (min < max)
    if x_min >= x_max or y_min >= y_max:
        return False
    
    return True


def merge_infographic_data(
    infographic_generated: List[Dict],
    color_idx: Dict,
    font_idx: Dict,
    start_wiki_idx: int = 0
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process infographic data with new format.
    
    Args:
        infographic_generated: List of infographic data with full_image_caption and background_caption
        color_idx: Color index mapping
        font_idx: Font index mapping
        start_wiki_idx: Starting wiki index for generating unique IDs
    
    Returns:
        Tuple of (easy_data, full_data) - easy version and full layout version
    """
    easy_result = []
    full_result = []
    
    for wiki_idx, infographic in enumerate(infographic_generated):
        # Generate unique wiki index based on start_wiki_idx
        wiki_id = start_wiki_idx + wiki_idx + 1  # Start from 1, not 0
        
        # Get the caption data from generated_infographic structure
        generated_infographic = infographic.get('generated_infographic', {})
        
        # Handle case where generated_infographic is a JSON string instead of dict
        if isinstance(generated_infographic, str):
            try:
                generated_infographic = json.loads(generated_infographic)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse generated_infographic for wiki {wiki_id:06d}, infographic_id: {infographic.get('infographic_id', 'unknown')}. Error: {e}")
                continue
        
        full_image_caption = generated_infographic.get('full_image_caption', '')
        background_caption = generated_infographic.get('background_caption', '')
        
        if not full_image_caption:
            print(f"Warning: No full_image_caption for wiki {wiki_id:06d}, infographic_id: {infographic.get('infographic_id', 'unknown')}")
            continue
        
        # Extract text elements with coordinates
        text_elements = extract_text_with_coordinates(full_image_caption)
        
        # Extract image elements with coordinates  
        image_elements = extract_images_with_coordinates(full_image_caption)
        
        # Select colors and font for this infographic (English only)
        num_colors = random.randint(1, 4)
        selected_colors = get_random_colors(color_idx, num_colors)
        selected_font = get_random_font(font_idx)
        selected_font_id = font_idx.get(selected_font, None)
        
        # Build layers_all for FULL output (complete layout)
        full_output_layers = []
        
        # Add the base layer with cleaned caption
        cleaned_caption = clean_caption_text(full_image_caption)
        base_layer = {
            'category': 'base',
            'top_left': [0, 0],  # Full canvas
            'bottom_right': [896, 2240],  # Full canvas size
            'caption': cleaned_caption
        }
        full_output_layers.append(base_layer)
        
        # Add background layer from background_caption (same as easy version)
        if background_caption:
            background_layer = {
                'category': 'element',
                'top_left': [0, 0],
                'bottom_right': [896, 2240],
                'caption': background_caption
            }
            full_output_layers.append(background_layer)
        
        # Add image elements as 'element' category
        for img_elem in image_elements:
            # Validate bounding box
            if not is_valid_bbox(img_elem['top_left'], img_elem['bottom_right']):
                print(f"Warning: Skipping invalid bbox for image in wiki {wiki_id:06d}: {img_elem['top_left']} - {img_elem['bottom_right']}")
                continue
            
            element_layer = {
                'category': 'element',
                'top_left': img_elem['top_left'],
                'bottom_right': img_elem['bottom_right'],
                'caption': img_elem['description']
            }
            full_output_layers.append(element_layer)
        
        # Add text elements as 'text' category
        for text_elem in text_elements:
            # Validate bounding box
            if not is_valid_bbox(text_elem['top_left'], text_elem['bottom_right']):
                print(f"Warning: Skipping invalid bbox for text in wiki {wiki_id:06d}: {text_elem['top_left']} - {text_elem['bottom_right']}")
                continue
            
            # Select random color for this text
            color = random.choice(selected_colors) if selected_colors else 'black'
            color_id = color_idx.get(color, 0)
            
            # Format caption with color and font
            if selected_font_id is None:
                # Fallback: try to map any en- font id; else default to 0
                try:
                    any_en_font_id = next(v for k, v in font_idx.items() if k.startswith('en-'))
                except StopIteration:
                    any_en_font_id = 0
                font_token = f'en-font-{any_en_font_id}'
            else:
                font_token = f'en-font-{selected_font_id}'

            caption = f'Text "{text_elem["text"]}" in <color-{color_id}>, <{font_token}>. '
            
            text_layer = {
                'category': 'text',
                'top_left': text_elem['top_left'],
                'bottom_right': text_elem['bottom_right'],
                'caption': caption,
                'text': text_elem['text']
            }
            full_output_layers.append(text_layer)
        
        # Create FULL layout version
        full_result_item = {
            'index': wiki_id,
            'layers_all': full_output_layers,
            'full_image_caption': cleaned_caption,
            'background_caption': background_caption
        }
        full_result.append(full_result_item)
        
        # Build layers_all for EASY output (base + background + 1 figure + title)
        easy_output_layers = []
        
        # Add base layer (same as full)
        easy_output_layers.append(base_layer.copy())
        
        # Add background layer from background_caption
        if background_caption:
            background_layer = {
                'category': 'element',
                'top_left': [0, 0],
                'bottom_right': [896, 2240],
                'caption': background_caption
            }
            easy_output_layers.append(background_layer)
        
        # Add 1 figure (first image element if available)
        if image_elements:
            # Find first valid image element
            for img_elem in image_elements:
                if is_valid_bbox(img_elem['top_left'], img_elem['bottom_right']):
                    figure_layer = {
                        'category': 'element',
                        'top_left': img_elem['top_left'],
                        'bottom_right': img_elem['bottom_right'],
                        'caption': img_elem['description']
                    }
                    easy_output_layers.append(figure_layer)
                    break  # Only add the first valid one
        
        # Add title (first text element if available)
        if text_elements:
            # Find first valid text element
            for text_elem in text_elements:
                if is_valid_bbox(text_elem['top_left'], text_elem['bottom_right']):
                    # Use same color and font selection as full version
                    color = selected_colors[0] if selected_colors else 'black'
                    color_id = color_idx.get(color, 0)
                    
                    if selected_font_id is None:
                        try:
                            any_en_font_id = next(v for k, v in font_idx.items() if k.startswith('en-'))
                        except StopIteration:
                            any_en_font_id = 0
                        font_token = f'en-font-{any_en_font_id}'
                    else:
                        font_token = f'en-font-{selected_font_id}'
                    
                    title_caption = f'Text "{text_elem["text"]}" in <color-{color_id}>, <{font_token}>. '
                    
                    title_layer = {
                        'category': 'text',
                        'top_left': text_elem['top_left'],
                        'bottom_right': text_elem['bottom_right'],
                        'caption': title_caption,
                        'text': text_elem['text']
                    }
                    easy_output_layers.append(title_layer)
                    break  # Only add the first valid one
        
        # Create EASY layout version
        easy_result_item = {
            'index': wiki_id,
            'layers_all': easy_output_layers,
            'full_image_caption': cleaned_caption,
            'background_caption': background_caption
        }
        easy_result.append(easy_result_item)
    
    return easy_result, full_result


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Process infographic data with new format (full_image_caption + background_caption)'
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
        default="src/data/create_data/output/infographic",
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
        help='Path to font_idx.json (default: ../bizgen/glyph/font_idx.json)'
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
        help='Base output directory (default: src/data/create_data/output) - will create bizgen_format_easy and bizgen_format_full subdirectories'
    )

    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths
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
    
    # Set base output directory
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        # Default to src/data/create_data/output from repository root
        repo_root = os.path.abspath(os.path.join(script_dir, '../../../../'))
        base_output_dir = os.path.join(repo_root, 'src/data/create_data/output')
    
    # Create easy and full output directories
    easy_output_dir = os.path.join(base_output_dir, 'bizgen_format_easy')
    full_output_dir = os.path.join(base_output_dir, 'bizgen_format_full')
    
    os.makedirs(easy_output_dir, exist_ok=True)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    print(f"  - Colors: {color_idx_path}")
    print(f"  - Fonts: {font_idx_path}")
    
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
    
    print(f"\nLoaded {len(color_idx)} colors")
    print(f"Loaded {len(font_idx)} fonts")
    
    # Process data
    print("\nProcessing data...")
    easy_data, full_data = merge_infographic_data(
        infographic_generated,
        color_idx,
        font_idx,
        start_wiki_idx=args.start_wiki
    )
    
    print(f"Generated {len(easy_data)} easy infographics")
    print(f"Generated {len(full_data)} full infographics")
    
    # Save results in chunks of 50 per file
    chunk_size = 50
    
    # Save EASY version
    print(f"\nSaving EASY version to {easy_output_dir}...")
    easy_saved_files = []
    
    for i in range(0, len(easy_data), chunk_size):
        chunk = easy_data[i:i + chunk_size]
        
        # Calculate file index based on first wiki ID in chunk
        first_wiki_id = chunk[0]['index']
        file_index = (first_wiki_id - 1) // chunk_size + 1  # Convert to 1-based file indexing
        
        filename = f"wiki{file_index:06d}.json"
        filepath = os.path.join(easy_output_dir, filename)
        
        save_json(chunk, filepath)
        easy_saved_files.append(filename)
        
        print(f"  Saved {len(chunk)} easy infographics to {filename} (wiki IDs: {chunk[0]['index']}-{chunk[-1]['index']})")
    
    # Save FULL version  
    print(f"\nSaving FULL version to {full_output_dir}...")
    full_saved_files = []
    
    for i in range(0, len(full_data), chunk_size):
        chunk = full_data[i:i + chunk_size]
        
        # Calculate file index based on first wiki ID in chunk
        first_wiki_id = chunk[0]['index']
        file_index = (first_wiki_id - 1) // chunk_size + 1  # Convert to 1-based file indexing
        
        filename = f"wiki{file_index:06d}.json"
        filepath = os.path.join(full_output_dir, filename)
        
        save_json(chunk, filepath)
        full_saved_files.append(filename)
        
        print(f"  Saved {len(chunk)} full infographics to {filename} (wiki IDs: {chunk[0]['index']}-{chunk[-1]['index']})")
    
    print(f"\nSaved {len(easy_saved_files)} easy files and {len(full_saved_files)} full files total")
    print("Done!")
    
    # Print summary statistics
    print("\n=== Summary ===")
    easy_total_layers = sum(len(item['layers_all']) for item in easy_data)
    full_total_layers = sum(len(item['layers_all']) for item in full_data)
    
    if easy_data:
        first_wiki_id = easy_data[0]['index']
        last_wiki_id = easy_data[-1]['index']
        print(f"Wiki ID range: {first_wiki_id:06d} - {last_wiki_id:06d}")
    
    print(f"Easy version:")
    print(f"  - Total infographics: {len(easy_data)}")
    print(f"  - Total files saved: {len(easy_saved_files)}")
    print(f"  - Total layers: {easy_total_layers}")
    print(f"  - Output directory: {easy_output_dir}")
    
    print(f"Full version:")
    print(f"  - Total infographics: {len(full_data)}")
    print(f"  - Total files saved: {len(full_saved_files)}")
    print(f"  - Total layers: {full_total_layers}")
    print(f"  - Output directory: {full_output_dir}")


if __name__ == '__main__':
    main()