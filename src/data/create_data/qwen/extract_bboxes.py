import json
import re
from pathlib import Path

def extract_bboxes_from_file(json_file_path):
    """
    Extract bounding boxes and categories from a JSON file.
    Also extracts the second element (index 1) as background.
    
    Args:
        json_file_path: Path to the JSON file
        
    Returns:
        Dictionary mapping source_index to list of items with their bbox data
    """
    print(f"Reading file: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Use dictionary to group by index
    grouped_data = {}
    total_bboxes = 0
    
    for item in data:
        index = item.get('index', 'unknown')
        layers_all = item.get('layers_all', [])
        
        if index not in grouped_data:
            grouped_data[index] = []
        
        for layer_idx, layer in enumerate(layers_all):
            # Check if this is the second element (index 1) with category "element"
            # and it should be a full-size background (896x2240)
            is_background = (
                layer_idx == 1 and 
                layer.get('category', '') == 'element' and
                layer.get('top_left', []) == [0, 0] and
                layer.get('bottom_right', []) == [896, 2240]
            )
            
            bbox_info = {
                'category': 'background' if is_background else layer.get('category', ''),
                'top_left': layer.get('top_left', []),
                'bottom_right': layer.get('bottom_right', [])
            }
            
            # Add caption if it's a background
            if is_background:
                bbox_info['caption'] = layer.get('caption', '')
            
            # Extract font and color information for text layers
            if layer.get('category', '') == 'text':
                caption = layer.get('caption', '')
                # Pattern to match: "in <color-X>, <font-info>."
                # This captures everything from "in" to the final period
                match = re.search(r'(in <color-\d+>, <[^>]+>\.)\s*$', caption)
                if match:
                    bbox_info['font_color_info'] = match.group(1)
            
            grouped_data[index].append(bbox_info)
            total_bboxes += 1
    
    print(f"  Extracted {total_bboxes} bounding boxes from {len(grouped_data)} items")
    return grouped_data


def main():
    # Define paths
    base_dir = Path("src/data/bizgen/meta")  # Directory containing the JSON files
    infographics_file = base_dir / 'infographics.json'
    multilang_file = base_dir / 'infographics_multilang.json'
    output_file = Path(__file__).parent / 'extracted_bboxes.json'
    
    print("=" * 60)
    print("Extracting Bounding Boxes from Infographics Files")
    print("=" * 60)
    
    # Dictionary to store all grouped data
    all_grouped_data = {}
    
    # Extract from infographics.json
    if infographics_file.exists():
        grouped_data = extract_bboxes_from_file(infographics_file)
        all_grouped_data.update(grouped_data)
    else:
        print(f"Warning: {infographics_file} not found!")
    
    # Extract from infographics_multilang.json
    if multilang_file.exists():
        grouped_data = extract_bboxes_from_file(multilang_file)
        all_grouped_data.update(grouped_data)
    else:
        print(f"Warning: {multilang_file} not found!")
    
    # Convert to list format for easier iteration
    output_data = []
    total_bboxes = 0
    
    for index, bboxes in all_grouped_data.items():
        output_data.append({
            'index': index,
            'bboxes': bboxes
        })
        total_bboxes += len(bboxes)
    
    # Save to output file
    print("\n" + "=" * 60)
    print(f"Saving {len(output_data)} items with {total_bboxes} total bounding boxes to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("Done!")
    print("=" * 60)
    
    # Print summary statistics
    from collections import Counter
    print("\nSummary:")
    print(f"  Total items: {len(output_data)}")
    print(f"  Total bounding boxes: {total_bboxes}")
    
    # Count by category
    category_counts = Counter()
    for item in output_data:
        for bbox in item['bboxes']:
            category_counts[bbox['category']] += 1
    
    print(f"\n  Categories:")
    for category, count in category_counts.items():
        print(f"  - {category}: {count} bboxes")


if __name__ == '__main__':
    main()

