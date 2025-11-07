#!/usr/bin/env python3
"""
Script to visualize all layouts from extracted_bboxes.json.
Shows bounding boxes with labels indicating category (element/text).
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse

INPUT_FILE = "/home/thinhnp/hf_vqa/src/data/narrator/extracted_bboxes.json"
OUTPUT_DIR = "/home/thinhnp/hf_vqa/src/data/narrator/bbox_visualizations"

# Color mapping for different categories
CATEGORY_COLORS = {
    'base': 'gray',
    'background': 'lightblue',
    'element': 'green',
    'text': 'red'
}


def visualize_layout(layout_data, output_path=None, show=True):
    """
    Visualize a single layout with all its bounding boxes.
    
    Args:
        layout_data: Dictionary containing layout info
        output_path: Path to save the visualization (optional)
        show: Whether to display the plot
    """
    index = layout_data['index']
    bboxes = layout_data['bboxes']
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 25))
    ax.set_xlim(0, 896)
    ax.set_ylim(2240, 0)  # Invert y-axis to match image coordinates
    ax.set_aspect('equal')
    ax.set_title(f'Layout {index} - Total bboxes: {len(bboxes)}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    
    # Draw each bbox
    for bbox_idx, bbox in enumerate(bboxes):
        category = bbox.get('category', 'unknown')
        top_left = bbox['top_left']
        bottom_right = bbox['bottom_right']
        
        # Calculate width and height
        x, y = top_left
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        area = width * height
        
        # Get color for this category
        color = CATEGORY_COLORS.get(category, 'orange')
        
        # Skip base category for cleaner visualization
        if category == 'base':
            continue
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add label with bbox index and category
        label_text = f'{bbox_idx}: {category}\n{area:,}px²'
        
        # Position label at top-left corner of bbox
        ax.text(
            x + 5, y + 20,
            label_text,
            fontsize=8,
            color=color,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8)
        )
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=color, linewidth=2, label=cat)
        for cat, color in CATEGORY_COLORS.items() if cat != 'base'
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_all_layouts(input_file, output_dir, start_idx=None, end_idx=None, show=False):
    """
    Visualize all layouts from the JSON file.
    
    Args:
        input_file: Path to extracted_bboxes.json
        output_dir: Directory to save visualizations
        start_idx: Start index (optional)
        end_idx: End index (optional)
        show: Whether to display plots interactively
    """
    # Load data
    print(f"Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        layouts = json.load(f)
    
    print(f"Total layouts: {len(layouts)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine range
    start = start_idx if start_idx is not None else 0
    end = end_idx if end_idx is not None else len(layouts)
    end = min(end, len(layouts))
    
    print(f"Visualizing layouts {start} to {end-1}")
    
    # Visualize each layout
    for i in range(start, end):
        layout = layouts[i]
        layout_idx = layout['index']
        
        # Format filename based on index type
        if isinstance(layout_idx, int):
            output_file = output_path / f"layout_{layout_idx:03d}.png"
        else:
            # For string indices, convert to safe filename
            try:
                layout_idx_num = int(layout_idx)
                output_file = output_path / f"layout_{layout_idx_num:03d}.png"
            except (ValueError, TypeError):
                # Use string as-is for filename
                safe_idx = str(layout_idx).replace('/', '_').replace('\\', '_')
                output_file = output_path / f"layout_{safe_idx}.png"
        
        print(f"Processing layout {layout['index']} ({i+1}/{end})...")
        visualize_layout(layout, output_path=output_file, show=show)
    
    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize bounding boxes from extracted_bboxes.json')
    parser.add_argument('--input', type=str, default=INPUT_FILE,
                        help='Input JSON file path')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output directory for visualizations')
    parser.add_argument('--start', type=int, default=None,
                        help='Start index (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                        help='End index (default: all)')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively (default: save only)')
    parser.add_argument('--single', type=int, default=None,
                        help='Visualize a single layout by index')
    
    args = parser.parse_args()
    
    if args.single is not None:
        # Visualize single layout
        with open(args.input, 'r', encoding='utf-8') as f:
            layouts = json.load(f)
        
        # Find layout with matching index
        layout = None
        for l in layouts:
            if l['index'] == args.single:
                layout = l
                break
        
        if layout:
            print(f"Visualizing layout {args.single}")
            visualize_layout(layout, output_path=None, show=True)
        else:
            print(f"Error: Layout with index {args.single} not found")
    else:
        # Visualize all or range
        visualize_all_layouts(
            args.input,
            args.output,
            start_idx=args.start,
            end_idx=args.end,
            show=args.show
        )


if __name__ == "__main__":
    main()

