import os
import json
import argparse
from typing import Dict, List, Any, Optional, Set


def load_json(filepath: str) -> Any:
    """Load JSON file safely."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def save_json(data: Any, filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_infographic_file_path(infographic_id: int, infographic_dir: str, offset: int = 0) -> str:
    """
    Get the path to the infographic JSON file containing data for the given infographic ID.
    Files are chunked into groups of 50.
    
    Args:
        infographic_id: The infographic ID to search for
        infographic_dir: Directory containing infographic files
        offset: ID offset to adjust file calculation
               Example: If ID 19030 should be at position 1 in file 000001, use offset=19029
    """
    # Adjust ID by offset
    adjusted_id = infographic_id - offset
    
    # Ensure adjusted_id is at least 1
    if adjusted_id < 1:
        adjusted_id = 1
    
    # Calculate which file chunk it belongs to
    file_index = (adjusted_id - 1) // 50 + 1
    filename = f"infographic{file_index:06d}.json"
    return os.path.join(infographic_dir, filename)


def find_infographic_entry_by_id(infographic_data: List[Dict], target_id: int) -> Optional[Dict]:
    """Find infographic entry with matching infographic_id."""
    for entry in infographic_data:
        if entry.get('infographic_id') == target_id:
            return entry
    return None


def load_infographic_entry(infographic_id: int, infographic_dir: str, offset: int = 0) -> Optional[Dict]:
    """
    Load a single infographic entry by ID from the infographic directory.
    
    Args:
        infographic_id: The infographic ID to load
        infographic_dir: Directory containing infographic JSON files
        offset: ID offset to adjust file calculation
        
    Returns:
        Infographic entry dict or None if not found
    """
    infographic_file_path = get_infographic_file_path(infographic_id, infographic_dir, offset)
    
    if not os.path.exists(infographic_file_path):
        print(f"  Warning: File not found: {infographic_file_path}")
        return None
    
    infographic_data = load_json(infographic_file_path)
    if not infographic_data:
        return None
    
    return find_infographic_entry_by_id(infographic_data, infographic_id)


def scan_wiki_files(wiki_dir: str, infographic_dir: str) -> Dict[str, List[int]]:
    """
    Scan all wiki*.json files and find problematic infographics.
    
    Args:
        wiki_dir: Directory containing wiki*.json files
        infographic_dir: Directory containing infographic*.json files (for loading entries)
        
    Returns:
        Dictionary with 'layout_86' and 'triple_backslash' lists of infographic IDs
    """
    problematic_ids = {
        'layout_86': [],
        'triple_backslash': []
    }
    
    if not os.path.exists(wiki_dir):
        print(f"Error: Wiki directory not found: {wiki_dir}")
        return problematic_ids
    
    # Get all wiki files
    wiki_files = sorted([f for f in os.listdir(wiki_dir) if f.startswith('wiki') and f.endswith('.json')])
    
    print(f"Scanning {len(wiki_files)} wiki files in {wiki_dir}...")
    
    total_entries = 0
    layout_86_count = 0
    triple_backslash_count = 0
    
    for wiki_file in wiki_files:
        filepath = os.path.join(wiki_dir, wiki_file)
        
        wiki_data = load_json(filepath)
        if not wiki_data or not isinstance(wiki_data, list):
            print(f"  Warning: Could not load {wiki_file}")
            continue
        
        for entry in wiki_data:
            total_entries += 1
            
            infographic_id = entry.get('index')
            if infographic_id is None:
                continue
            
            # Check 1: Layout index 86
            original_bbox_index = entry.get('original_bbox_index')
            if original_bbox_index == 86:
                problematic_ids['layout_86'].append(infographic_id)
                layout_86_count += 1
            
            # Check 2: Triple backslash in full_image_caption
            # Pattern: \\\" (backslash backslash quote) indicates escaped quotes issue
            # This appears as \\" in the decoded JSON string
            # full_image_caption = entry.get('full_image_caption', '')
            # if '\\"' in full_image_caption or '\\\\' in full_image_caption:
            #     problematic_ids['triple_backslash'].append(infographic_id)
            #     triple_backslash_count += 1
    
    print(f"\nScan complete!")
    print(f"  Total entries scanned: {total_entries}")
    print(f"  Layout 86 issues: {layout_86_count}")
    # print(f"  Triple backslash issues: {triple_backslash_count}")
    
    # Remove duplicates and sort
    problematic_ids['layout_86'] = sorted(list(set(problematic_ids['layout_86'])))
    # problematic_ids['triple_backslash'] = sorted(list(set(problematic_ids['triple_backslash'])))
    
    return problematic_ids


def append_to_failed_json(
    problematic_ids: Dict[str, List[int]],
    infographic_dir: str,
    output_file: str,
    offset: int = 0
) -> int:
    """
    Append problematic infographic entries to failed.json.
    Skip IDs that already exist in the file.
    
    Args:
        problematic_ids: Dict with lists of problematic IDs by category
        infographic_dir: Directory containing infographic*.json files
        output_file: Path to failed.json
        offset: ID offset to adjust file calculation
        
    Returns:
        Number of new entries added
    """
    # Load existing failed.json if it exists
    existing_entries = []
    existing_ids: Set[int] = set()
    
    if os.path.exists(output_file):
        existing_data = load_json(output_file)
        if existing_data and isinstance(existing_data, list):
            existing_entries = existing_data
            existing_ids = {entry.get('infographic_id') for entry in existing_entries if entry.get('infographic_id')}
            print(f"\nLoaded {len(existing_entries)} existing entries from {output_file}")
        else:
            print(f"\nExisting file is not a valid list, starting fresh")
    else:
        print(f"\nNo existing failed.json found, creating new file")
    
    # Collect all unique problematic IDs
    all_problematic_ids = set()
    for category, ids in problematic_ids.items():
        all_problematic_ids.update(ids)
    
    all_problematic_ids = sorted(list(all_problematic_ids))
    
    print(f"\nTotal unique problematic IDs found: {len(all_problematic_ids)}")
    print(f"  Already in failed.json: {len([id for id in all_problematic_ids if id in existing_ids])}")
    print(f"  New IDs to add: {len([id for id in all_problematic_ids if id not in existing_ids])}")
    
    # Load and append new entries
    new_entries_added = 0
    
    for infographic_id in all_problematic_ids:
        # Skip if already exists
        if infographic_id in existing_ids:
            continue
        
        # Load the infographic entry with offset
        entry = load_infographic_entry(infographic_id, infographic_dir, offset)
        
        if entry:
            existing_entries.append(entry)
            new_entries_added += 1
            
            # Determine which categories this ID belongs to
            categories = []
            if infographic_id in problematic_ids['layout_86']:
                categories.append('layout_86')
            if infographic_id in problematic_ids['triple_backslash']:
                categories.append('triple_backslash')
            
            print(f"  Added ID {infographic_id} ({', '.join(categories)})")
        else:
            print(f"  Warning: Could not load entry for ID {infographic_id}")
    
    # Save updated failed.json
    save_json(existing_entries, output_file)
    
    print(f"\n✓ Saved {len(existing_entries)} total entries to {output_file}")
    print(f"  ({new_entries_added} new entries added)")
    
    return new_entries_added


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Find and append problematic infographic entries to failed.json'
    )
    parser.add_argument(
        '--wiki-dir',
        type=str,
        default='/home/thinhnp/hf_vqa/src/data/narrator/wiki_val',
        help='Directory containing wiki*.json files (default: /home/thinhnp/hf_vqa/src/data/narrator/wiki_val)'
    )
    parser.add_argument(
        '--infographic-dir',
        type=str,
        default='/home/thinhnp/hf_vqa/src/data/narrator/infographic_val',
        help='Directory containing infographic*.json files (default: /home/thinhnp/hf_vqa/src/data/narrator/infographic_val)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='/home/thinhnp/hf_vqa/src/data/narrator/infographic_val/failed.json',
        help='Output path for failed.json (default: /home/thinhnp/hf_vqa/src/data/narrator/infographic_val/failed.json)'
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=19029,
        help='ID offset for file calculation. Example: If ID 19030 is at position 1 in file 000001, use --offset 19029. Default: 19029 for val set'
    )
    
    args = parser.parse_args()
    
    print("=== Find Failed Infographics ===")
    print(f"Wiki directory: {args.wiki_dir}")
    print(f"Infographic directory: {args.infographic_dir}")
    print(f"Output file: {args.output_file}")
    print()
    
    # Validate directories
    if not os.path.exists(args.wiki_dir):
        print(f"Error: Wiki directory not found: {args.wiki_dir}")
        return
    
    if not os.path.exists(args.infographic_dir):
        print(f"Error: Infographic directory not found: {args.infographic_dir}")
        return
    
    # Scan wiki files for problematic entries
    print("=== Phase 1: Scanning wiki files ===")
    print(f"Using ID offset: {args.offset}")
    problematic_ids = scan_wiki_files(args.wiki_dir, args.infographic_dir)
    
    # Display findings
    print("\n=== Findings ===")
    print(f"Layout 86 issues: {len(problematic_ids['layout_86'])} IDs")
    if problematic_ids['layout_86']:
        print(f"  Sample IDs: {problematic_ids['layout_86'][:10]}")
        if len(problematic_ids['layout_86']) > 10:
            print(f"  ... and {len(problematic_ids['layout_86']) - 10} more")
    
    print(f"\nTriple backslash issues: {len(problematic_ids['triple_backslash'])} IDs")
    if problematic_ids['triple_backslash']:
        print(f"  Sample IDs: {problematic_ids['triple_backslash'][:10]}")
        if len(problematic_ids['triple_backslash']) > 10:
            print(f"  ... and {len(problematic_ids['triple_backslash']) - 10} more")
    
    # Append to failed.json
    print("\n=== Phase 2: Updating failed.json ===")
    new_count = append_to_failed_json(
        problematic_ids,
        args.infographic_dir,
        args.output_file,
        offset=args.offset
    )
    
    print("\n=== Summary ===")
    print(f"Scan completed successfully!")
    print(f"  Layout 86 issues found: {len(problematic_ids['layout_86'])}")
    print(f"  Triple backslash issues found: {len(problematic_ids['triple_backslash'])}")
    print(f"  New entries added to failed.json: {new_count}")
    print(f"  Output file: {args.output_file}")


if __name__ == '__main__':
    main()
