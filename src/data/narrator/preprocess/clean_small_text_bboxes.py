import json
from typing import List, Dict

AREA_THRESHOLD = 15000
INPUT_FILE = "/home/thinhnp/hf_vqa/src/data/narrator/extracted_bboxes.json"


def calculate_bbox_area(bbox: Dict) -> int:
    """Calculate area of a bounding box."""
    top_left = bbox['top_left']
    bottom_right = bbox['bottom_right']
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    return width * height


def filter_small_text_bboxes(layouts: List[Dict], min_area: int = AREA_THRESHOLD) -> tuple[List[Dict], int]:
    """
    Filter out text bboxes with area less than min_area.
    
    Args:
        layouts: List of layout dictionaries containing bboxes
        min_area: Minimum area threshold in pixels^2
        
    Returns:
        Tuple of (filtered_layouts, total_removed_count)
    """
    total_removed = 0
    filtered_layouts = []
    
    for layout in layouts:
        filtered_bboxes = []
        layout_removed = 0
        
        for bbox in layout.get('bboxes', []):
            category = bbox.get('category', '')
            
            # Keep all non-text bboxes
            if category != 'text':
                filtered_bboxes.append(bbox)
            else:
                # For text bboxes, check area threshold
                area = calculate_bbox_area(bbox)
                if area >= min_area:
                    filtered_bboxes.append(bbox)
                else:
                    layout_removed += 1
        
        total_removed += layout_removed
        
        # Create new layout with filtered bboxes
        filtered_layout = layout.copy()
        filtered_layout['bboxes'] = filtered_bboxes
        filtered_layouts.append(filtered_layout)
    
    return filtered_layouts, total_removed


def main():
    """Main function to load, filter, and save the extracted_bboxes.json file."""
    print(f"Loading data from: {INPUT_FILE}")
    
    # Load the JSON file
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        layouts = json.load(f)
    
    print(f"Loaded {len(layouts)} layouts")
    
    # Count original text bboxes
    original_text_count = 0
    for layout in layouts:
        for bbox in layout.get('bboxes', []):
            if bbox.get('category') == 'text':
                original_text_count += 1
    
    print(f"Original text bboxes count: {original_text_count}")
    print(f"Filtering text bboxes with area < {AREA_THRESHOLD} pixels^2...")
    
    # Filter small text bboxes
    filtered_layouts, removed_count = filter_small_text_bboxes(layouts, AREA_THRESHOLD)
    
    # Count remaining text bboxes
    remaining_text_count = 0
    for layout in filtered_layouts:
        for bbox in layout.get('bboxes', []):
            if bbox.get('category') == 'text':
                remaining_text_count += 1
    
    print(f"\nResults:")
    print(f"  - Total text bboxes removed: {removed_count}")
    print(f"  - Remaining text bboxes: {remaining_text_count}")
    print(f"  - Percentage removed: {removed_count / original_text_count * 100:.2f}%")
    
    # Save the filtered data back to the original file
    print(f"\nSaving filtered data to: {INPUT_FILE}")
    with open(INPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_layouts, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()

