import json
import argparse
from pathlib import Path

INPUT_FILE = "/home/thinhnp/hf_vqa/src/data/narrator/extracted_bboxes.json"


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filepath):
    """Save JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def find_layout_by_index(layouts, layout_index):
    """Find layout by its index field."""
    for i, layout in enumerate(layouts):
        if layout['index'] == layout_index:
            return i, layout
    return None, None


def get_font_color_from_layout(layout):
    """
    Extract font_color_info from existing text bboxes in the layout.
    
    Args:
        layout: Layout dictionary containing bboxes
        
    Returns:
        font_color_info string, or None if no text bboxes found
    """
    for bbox in layout.get('bboxes', []):
        if bbox.get('category') == 'text' and 'font_color_info' in bbox:
            return bbox['font_color_info']
    return None


def toggle_bbox_category(input_file, layout_index, bbox_index, dry_run=False):
    """
    Toggle bbox category between 'text' and 'element'.
    When converting element to text, automatically adds font_color_info from other text bboxes.
    
    Args:
        input_file: Path to extracted_bboxes.json
        layout_index: Index of the layout
        bbox_index: Index of the bbox within the layout
        dry_run: If True, only show what would be changed without saving
        
    Returns:
        True if successful, False otherwise
    """
    # Load data
    print(f"Loading data from: {input_file}")
    layouts = load_json(input_file)
    print(f"Total layouts: {len(layouts)}")
    
    # Find the layout
    list_idx, layout = find_layout_by_index(layouts, layout_index)
    
    if layout is None:
        print(f"Error: Layout with index {layout_index} not found!")
        return False
    
    print(f"Found layout at position {list_idx} with {len(layout['bboxes'])} bboxes")
    
    # Check if bbox_index is valid
    if bbox_index < 0 or bbox_index >= len(layout['bboxes']):
        print(f"Error: Bbox index {bbox_index} is out of range (0-{len(layout['bboxes'])-1})")
        return False
    
    # Get the bbox
    bbox = layout['bboxes'][bbox_index]
    old_category = bbox.get('category', 'unknown')
    
    # Determine new category
    if old_category == 'text':
        new_category = 'element'
    elif old_category == 'element':
        new_category = 'text'
    else:
        print(f"Error: Cannot toggle category '{old_category}' (only 'text' and 'element' can be toggled)")
        return False
    
    # Calculate bbox info for display
    top_left = bbox['top_left']
    bottom_right = bbox['bottom_right']
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    area = width * height
    
    # If converting element to text, get font_color_info from other text bboxes
    font_color_info = None
    if new_category == 'text':
        font_color_info = get_font_color_from_layout(layout)
        if font_color_info:
            print(f"  Font/Color Info: {font_color_info} (from other text bboxes in layout)")
        else:
            print(f"  Warning: No text bboxes found in layout to copy font_color_info from")
    
    # Display change info
    print(f"\nBbox Information:")
    print(f"  Layout Index: {layout_index}")
    print(f"  Bbox Index: {bbox_index}")
    print(f"  Position: {top_left} to {bottom_right}")
    print(f"  Size: {width}x{height} ({area:,} pixels²)")
    print(f"  Current Category: {old_category}")
    print(f"  New Category: {new_category}")
    
    if dry_run:
        print("\nDRY RUN - No changes saved")
        return True
    
    # Apply the change
    bbox['category'] = new_category
    
    # Add or remove font_color_info based on new category
    if new_category == 'text' and font_color_info:
        bbox['font_color_info'] = font_color_info
        print(f"  Added font_color_info: {font_color_info}")
    elif new_category == 'element' and 'font_color_info' in bbox:
        del bbox['font_color_info']
        print(f"  Removed font_color_info")
    
    layouts[list_idx]['bboxes'][bbox_index] = bbox
    
    # Save back to file
    print(f"\nSaving changes to: {input_file}")
    save_json(layouts, input_file)
    print("Successfully toggled bbox category!")
    
    return True


def delete_bbox(input_file, layout_index, bbox_index, dry_run=False):
    """
    Delete a specific bbox from a layout.
    
    Args:
        input_file: Path to extracted_bboxes.json
        layout_index: Index of the layout
        bbox_index: Index of the bbox to delete
        dry_run: If True, only show what would be deleted without saving
        
    Returns:
        True if successful, False otherwise
    """
    # Load data
    print(f"Loading data from: {input_file}")
    layouts = load_json(input_file)
    print(f"Total layouts: {len(layouts)}")
    
    # Find the layout
    list_idx, layout = find_layout_by_index(layouts, layout_index)
    
    if layout is None:
        print(f"Error: Layout with index {layout_index} not found!")
        return False
    
    print(f"Found layout at position {list_idx} with {len(layout['bboxes'])} bboxes")
    
    # Check if bbox_index is valid
    if bbox_index < 0 or bbox_index >= len(layout['bboxes']):
        print(f"Error: Bbox index {bbox_index} is out of range (0-{len(layout['bboxes'])-1})")
        return False
    
    # Get bbox info for display
    bbox = layout['bboxes'][bbox_index]
    category = bbox.get('category', 'unknown')
    top_left = bbox['top_left']
    bottom_right = bbox['bottom_right']
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    area = width * height
    
    # Display bbox info
    print(f"\nBbox to be deleted:")
    print(f"  Layout Index: {layout_index}")
    print(f"  Bbox Index: {bbox_index}")
    print(f"  Category: {category}")
    print(f"  Position: {top_left} to {bottom_right}")
    print(f"  Size: {width}x{height} ({area:,} pixels²)")
    
    if dry_run:
        print("\nDRY RUN - No changes saved")
        return True
    
    # Delete the bbox
    del layouts[list_idx]['bboxes'][bbox_index]
    
    # Save back to file
    print(f"\nSaving changes to: {input_file}")
    save_json(layouts, input_file)
    print(f"Successfully deleted bbox {bbox_index} from layout {layout_index}!")
    print(f"Remaining bboxes in layout: {len(layouts[list_idx]['bboxes'])}")
    
    return True


def delete_layout(input_file, layout_index, dry_run=False):
    """
    Delete an entire layout from the JSON file.
    
    Args:
        input_file: Path to extracted_bboxes.json
        layout_index: Index of the layout to delete
        dry_run: If True, only show what would be deleted without saving
        
    Returns:
        True if successful, False otherwise
    """
    # Load data
    print(f"Loading data from: {input_file}")
    layouts = load_json(input_file)
    print(f"Total layouts: {len(layouts)}")
    
    # Find the layout
    list_idx, layout = find_layout_by_index(layouts, layout_index)
    
    if layout is None:
        print(f"Error: Layout with index {layout_index} not found!")
        return False
    
    # Display layout info
    print(f"\nLayout to be deleted:")
    print(f"  Layout Index: {layout_index}")
    print(f"  Position in list: {list_idx}")
    print(f"  Total bboxes: {len(layout.get('bboxes', []))}")
    
    if dry_run:
        print("\nDRY RUN - No changes saved")
        return True
    
    # Delete the layout
    del layouts[list_idx]
    
    # Save back to file
    print(f"\nSaving changes to: {input_file}")
    save_json(layouts, input_file)
    print(f"Successfully deleted layout {layout_index}!")
    print(f"New total layouts: {len(layouts)}")
    
    return True


def list_layout_bboxes(input_file, layout_index):
    """
    List all bboxes in a specific layout.
    
    Args:
        input_file: Path to extracted_bboxes.json
        layout_index: Index of the layout
    """
    # Load data
    layouts = load_json(input_file)
    
    # Find the layout
    list_idx, layout = find_layout_by_index(layouts, layout_index)
    
    if layout is None:
        print(f"Error: Layout with index {layout_index} not found!")
        return
    
    print(f"\nLayout {layout_index} - Total bboxes: {len(layout['bboxes'])}")
    print("-" * 80)
    
    # List all bboxes
    for i, bbox in enumerate(layout['bboxes']):
        category = bbox.get('category', 'unknown')
        top_left = bbox['top_left']
        bottom_right = bbox['bottom_right']
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        area = width * height
        
        print(f"[{i:2d}] {category:12s} | {top_left} -> {bottom_right} | {width:4d}x{height:4d} | {area:8,} px²")


def main():
    parser = argparse.ArgumentParser(
        description='Toggle bbox category between text and element, or delete layout/bbox',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all bboxes in layout 0
  python toggle_bbox_category.py --layout 0 --list
  
  # Toggle single bbox 5 in layout 0 (dry run)
  python toggle_bbox_category.py --layout 0 --bbox 5 --dry-run
  
  # Toggle single bbox 5 in layout 0 (actual change)
  python toggle_bbox_category.py --layout 0 --bbox 5
  
  # Toggle multiple bboxes (11, 6, 12) in layout 18
  python toggle_bbox_category.py --layout 18 --bbox 11 6 12
  
  # Delete single bbox 12 from layout 11 (dry run)
  python toggle_bbox_category.py --layout 11 --bbox 12 --delete --dry-run
  
  # Delete multiple bboxes from layout 11
  python toggle_bbox_category.py --layout 11 --bbox 12 10 8 --delete
  
  # Delete entire layout 0 (dry run)
  python toggle_bbox_category.py --layout 0 --delete --dry-run
  
  # Delete entire layout 0 (actual deletion)
  python toggle_bbox_category.py --layout 0 --delete
        """
    )
    
    parser.add_argument('--input', type=str, default=INPUT_FILE,
                        help='Input JSON file path')
    parser.add_argument('--layout', type=int, required=True,
                        help='Layout index')
    parser.add_argument('--bbox', type=int, nargs='+', default=None,
                        help='Bbox index(es) to toggle or delete (can specify multiple)')
    parser.add_argument('--list', action='store_true',
                        help='List all bboxes in the layout')
    parser.add_argument('--delete', action='store_true',
                        help='Delete the bbox (if --bbox specified) or entire layout (if no --bbox)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be changed without saving')
    
    args = parser.parse_args()
    
    if args.list:
        # List mode
        list_layout_bboxes(args.input, args.layout)
    elif args.delete and args.bbox is not None:
        # Delete specific bbox(es) mode
        bbox_indices = args.bbox if isinstance(args.bbox, list) else [args.bbox]
        # Sort in descending order to avoid index shift when deleting
        bbox_indices_sorted = sorted(bbox_indices, reverse=True)
        
        print(f"Processing {len(bbox_indices_sorted)} bbox(es) for deletion...")
        for bbox_idx in bbox_indices_sorted:
            print(f"\n{'='*80}")
            success = delete_bbox(args.input, args.layout, bbox_idx, dry_run=args.dry_run)
            if not success:
                print(f"Failed to delete bbox {bbox_idx}")
                break
        print(f"\n{'='*80}")
        print(f"Completed processing {len(bbox_indices_sorted)} bbox(es)")
        
    elif args.delete:
        # Delete entire layout mode
        delete_layout(args.input, args.layout, dry_run=args.dry_run)
    elif args.bbox is not None:
        # Toggle mode (multiple bboxes)
        bbox_indices = args.bbox if isinstance(args.bbox, list) else [args.bbox]
        
        print(f"Processing {len(bbox_indices)} bbox(es) for toggle...")
        for bbox_idx in bbox_indices:
            print(f"\n{'='*80}")
            success = toggle_bbox_category(args.input, args.layout, bbox_idx, dry_run=args.dry_run)
            if not success:
                print(f"Failed to toggle bbox {bbox_idx}")
                break
        print(f"\n{'='*80}")
        print(f"Completed processing {len(bbox_indices)} bbox(es)")
    else:
        print("Error: Either --list, --delete, or --bbox must be specified")
        parser.print_help()


if __name__ == "__main__":
    main()

