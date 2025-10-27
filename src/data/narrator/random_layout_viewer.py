#!/usr/bin/env python3
"""
Random Layout Viewer for Extracted BBoxes

This script loads the extracted_bboxes.json file and provides functionality
to randomly select and display layout information in a user-friendly format.

Usage:
    python random_layout_viewer.py [--index INDEX] [--count COUNT] [--categories CATEGORIES]
    
Examples:
    python random_layout_viewer.py                    # Show 1 random layout
    python random_layout_viewer.py --count 3          # Show 3 random layouts
    python random_layout_viewer.py --index 5          # Show specific layout by index
    python random_layout_viewer.py --categories element,background  # Filter by categories
"""

import json
import random
import argparse
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class LayoutViewer:
    """Class to handle layout viewing and analysis"""
    
    def __init__(self, bboxes_file: str = "extracted_bboxes.json"):
        """Initialize with bboxes file path"""
        self.bboxes_file = Path(__file__).parent / bboxes_file
        self.layouts = []
        self.load_layouts()
    
    def load_layouts(self):
        """Load layouts from JSON file"""
        try:
            with open(self.bboxes_file, 'r', encoding='utf-8') as f:
                self.layouts = json.load(f)
            print(f"✅ Loaded {len(self.layouts)} layouts from {self.bboxes_file}")
        except FileNotFoundError:
            print(f"❌ Error: File {self.bboxes_file} not found!")
            exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON: {e}")
            exit(1)
    
    def get_layout_stats(self) -> Dict[str, Any]:
        """Get statistics about all layouts"""
        if not self.layouts:
            return {}
        
        total_layouts = len(self.layouts)
        all_categories = []
        bbox_counts = []
        
        for layout in self.layouts:
            bboxes = layout.get('bboxes', [])
            bbox_counts.append(len(bboxes))
            for bbox in bboxes:
                all_categories.append(bbox.get('category', 'unknown'))
        
        category_counts = {}
        for cat in all_categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            'total_layouts': total_layouts,
            'avg_bboxes_per_layout': sum(bbox_counts) / len(bbox_counts) if bbox_counts else 0,
            'min_bboxes': min(bbox_counts) if bbox_counts else 0,
            'max_bboxes': max(bbox_counts) if bbox_counts else 0,
            'category_distribution': category_counts,
            'available_indices': [layout.get('index', i) for i, layout in enumerate(self.layouts)]
        }
    
    def get_random_layout(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get a random layout, optionally filtered by categories"""
        if not self.layouts:
            return {}
        
        if categories:
            # Filter layouts that have at least one bbox with specified categories
            filtered_layouts = []
            for layout in self.layouts:
                bboxes = layout.get('bboxes', [])
                if any(bbox.get('category') in categories for bbox in bboxes):
                    filtered_layouts.append(layout)
            
            if not filtered_layouts:
                print(f"⚠️  No layouts found with categories: {categories}")
                return {}
            
            return random.choice(filtered_layouts)
        
        return random.choice(self.layouts)
    
    def get_layout_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get layout by specific index"""
        for layout in self.layouts:
            if layout.get('index') == index:
                return layout
        return None
    
    def format_bbox(self, bbox: Dict[str, Any], bbox_id: int) -> str:
        """Format a single bbox for display"""
        category = bbox.get('category', 'unknown')
        top_left = bbox.get('top_left', [0, 0])
        bottom_right = bbox.get('bottom_right', [0, 0])
        
        # Calculate dimensions
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        area = width * height
        
        # Format basic info
        result = f"    [{bbox_id:2d}] {category.upper():<12} | "
        result += f"Position: ({top_left[0]:4d}, {top_left[1]:4d}) → ({bottom_right[0]:4d}, {bottom_right[1]:4d}) | "
        result += f"Size: {width:4d}×{height:4d} | Area: {area:7d}"
        
        # Add caption if available
        caption = bbox.get('caption', '').strip()
        if caption:
            # Truncate long captions
            if len(caption) > 80:
                caption = caption[:77] + "..."
            result += f"\n        Caption: {caption}"
        
        return result
    
    def display_layout(self, layout: Dict[str, Any], show_details: bool = True):
        """Display a layout in formatted way"""
        if not layout:
            print("❌ No layout to display")
            return
        
        # Header
        layout_index = layout.get('index', 'Unknown')
        bboxes = layout.get('bboxes', [])
        
        print("\n" + "="*100)
        print(f"📊 LAYOUT #{layout_index}")
        print("="*100)
        
        if not bboxes:
            print("⚠️  No bboxes found in this layout")
            return
        
        # Summary
        categories = {}
        for bbox in bboxes:
            cat = bbox.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"📈 Summary: {len(bboxes)} bboxes total")
        for cat, count in sorted(categories.items()):
            print(f"   • {cat}: {count}")
        print()
        
        if show_details:
            print("📋 Detailed BBox Information:")
            print("    ID  Category     | Position (x1, y1) → (x2, y2)        | Dimensions  | Area")
            print("    " + "-"*90)
            
            # Sort bboxes by category for better readability
            sorted_bboxes = sorted(bboxes, key=lambda x: x.get('category', 'zzz'))
            
            for i, bbox in enumerate(sorted_bboxes, 1):
                print(self.format_bbox(bbox, i))
        
        print("="*100)
    
    def display_multiple_layouts(self, count: int = 1, categories: Optional[List[str]] = None, 
                               show_details: bool = True):
        """Display multiple random layouts"""
        print(f"\n🎲 Displaying {count} random layout(s)")
        if categories:
            print(f"🏷️  Filtered by categories: {categories}")
        
        displayed = 0
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loop
        
        while displayed < count and attempts < max_attempts:
            attempts += 1
            layout = self.get_random_layout(categories)
            if layout:
                self.display_layout(layout, show_details)
                displayed += 1
            else:
                break
        
        if displayed < count:
            print(f"⚠️  Only found {displayed} layout(s) matching criteria")
    
    def show_stats(self):
        """Display overall statistics"""
        stats = self.get_layout_stats()
        if not stats:
            print("❌ No statistics available")
            return
        
        print("\n" + "="*60)
        print("📊 LAYOUT COLLECTION STATISTICS")
        print("="*60)
        print(f"Total layouts: {stats['total_layouts']}")
        print(f"Average bboxes per layout: {stats['avg_bboxes_per_layout']:.1f}")
        print(f"BBox count range: {stats['min_bboxes']} - {stats['max_bboxes']}")
        print()
        print("📊 Category Distribution:")
        for cat, count in sorted(stats['category_distribution'].items(), key=lambda x: -x[1]):
            percentage = (count / sum(stats['category_distribution'].values())) * 100
            print(f"   • {cat:<15}: {count:4d} ({percentage:5.1f}%)")
        print()
        print(f"📋 Available indices: {stats['available_indices'][:10]}{'...' if len(stats['available_indices']) > 10 else ''}")
        print("="*60)
    
    def output_layout_as_json(self, layout: Dict[str, Any]):
        """Output a layout as clean JSON object to stdout"""
        if not layout:
            print(json.dumps({"error": "No layout to display"}, indent=2))
            return
        
        # Create clean output structure
        output = {
            "layout_index": layout.get('index', 'Unknown'),
            "total_bboxes": len(layout.get('bboxes', [])),
            "bboxes": []
        }
        
        # Process bboxes
        for i, bbox in enumerate(layout.get('bboxes', []), 1):
            top_left = bbox.get('top_left', [0, 0])
            bottom_right = bbox.get('bottom_right', [0, 0])
            
            bbox_info = {
                "id": i,
                "category": bbox.get('category', 'unknown'),
                "position": {
                    "top_left": top_left,
                    "bottom_right": bottom_right
                },
                "dimensions": {
                    "width": bottom_right[0] - top_left[0],
                    "height": bottom_right[1] - top_left[1],
                    "area": (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])
                }
            }
            
            # Add caption if available
            caption = bbox.get('caption', '').strip()
            if caption:
                bbox_info["caption"] = caption
            
            output["bboxes"].append(bbox_info)
        
        # Add category summary
        categories = {}
        for bbox in layout.get('bboxes', []):
            cat = bbox.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        output["category_summary"] = categories
        
        # Print clean JSON
        print(json.dumps(output, indent=2, ensure_ascii=False))
    
    def output_multiple_layouts_as_json(self, count: int = 1, categories: Optional[List[str]] = None):
        """Output multiple layouts as JSON array"""
        layouts_output = []
        
        displayed = 0
        attempts = 0
        max_attempts = count * 10
        
        while displayed < count and attempts < max_attempts:
            attempts += 1
            layout = self.get_random_layout(categories)
            if layout:
                # Create layout object without printing
                layout_obj = {
                    "layout_index": layout.get('index', 'Unknown'),
                    "total_bboxes": len(layout.get('bboxes', [])),
                    "bboxes": []
                }
                
                # Process bboxes
                for i, bbox in enumerate(layout.get('bboxes', []), 1):
                    top_left = bbox.get('top_left', [0, 0])
                    bottom_right = bbox.get('bottom_right', [0, 0])
                    
                    bbox_info = {
                        "id": i,
                        "category": bbox.get('category', 'unknown'),
                        "position": {
                            "top_left": top_left,
                            "bottom_right": bottom_right
                        },
                        "dimensions": {
                            "width": bottom_right[0] - top_left[0],
                            "height": bottom_right[1] - top_left[1],
                            "area": (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])
                        }
                    }
                    
                    caption = bbox.get('caption', '').strip()
                    if caption:
                        bbox_info["caption"] = caption
                    
                    layout_obj["bboxes"].append(bbox_info)
                
                # Add category summary
                categories_count = {}
                for bbox in layout.get('bboxes', []):
                    cat = bbox.get('category', 'unknown')
                    categories_count[cat] = categories_count.get(cat, 0) + 1
                
                layout_obj["category_summary"] = categories_count
                layouts_output.append(layout_obj)
                displayed += 1
            else:
                break
        
        # Create final output
        final_output = {
            "requested_count": count,
            "actual_count": displayed,
            "filter_categories": categories,
            "layouts": layouts_output
        }
        
        print(json.dumps(final_output, indent=2, ensure_ascii=False))


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Random Layout Viewer for Extracted BBoxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python random_layout_viewer.py                           # Show 1 random layout
  python random_layout_viewer.py --count 3                 # Show 3 random layouts  
  python random_layout_viewer.py --index 5                 # Show specific layout
  python random_layout_viewer.py --categories element      # Filter by category
  python random_layout_viewer.py --stats                   # Show statistics only
  python random_layout_viewer.py --count 2 --no-details    # Show without details
  python random_layout_viewer.py --json                    # Output as JSON for ChatGPT
  python random_layout_viewer.py --index 5 --json          # Output specific layout as JSON
        """
    )
    
    parser.add_argument(
        '--index', 
        type=int, 
        help='Show specific layout by index'
    )
    
    parser.add_argument(
        '--count', 
        type=int, 
        default=1,
        help='Number of random layouts to display (default: 1)'
    )
    
    parser.add_argument(
        '--categories', 
        type=str,
        help='Comma-separated list of categories to filter by (e.g., element,background)'
    )
    
    parser.add_argument(
        '--stats', 
        action='store_true',
        help='Show statistics only (no layout display)'
    )
    
    parser.add_argument(
        '--no-details', 
        action='store_true',
        help='Show summary only, without detailed bbox information'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        default='extracted_bboxes.json',
        help='Path to extracted bboxes JSON file (default: extracted_bboxes.json)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output layout(s) as clean JSON object to stdout (for ChatGPT input)'
    )
    
    args = parser.parse_args()
    
    # Initialize viewer
    viewer = LayoutViewer(args.file)
    
    # Parse categories if provided
    categories = None
    if args.categories:
        categories = [cat.strip() for cat in args.categories.split(',')]
    
    # Show stats if requested
    if args.stats:
        viewer.show_stats()
        return
    
    # Show specific layout by index
    if args.index is not None:
        layout = viewer.get_layout_by_index(args.index)
        if layout:
            if args.json:
                viewer.output_layout_as_json(layout)
            else:
                print(f"🎯 Showing layout with index {args.index}")
                viewer.display_layout(layout, not args.no_details)
        else:
            if args.json:
                print(json.dumps({"error": f"Layout with index {args.index} not found"}, indent=2))
            else:
                print(f"❌ Layout with index {args.index} not found")
                stats = viewer.get_layout_stats()
                available = stats.get('available_indices', [])
                print(f"Available indices: {available[:20]}{'...' if len(available) > 20 else ''}")
        return
    
    # Show random layouts
    if args.json:
        viewer.output_multiple_layouts_as_json(
            count=args.count, 
            categories=categories
        )
    else:
        viewer.display_multiple_layouts(
            count=args.count, 
            categories=categories, 
            show_details=not args.no_details
        )
        
        # Show mini stats at the end
        if args.count == 1 and not args.no_details:
            print(f"\n💡 Tip: Use --stats to see collection overview, or --count N for multiple layouts")


if __name__ == "__main__":
    main()