#!/usr/bin/env python3
"""
Script to extract a specific infographic entry by ID and save it to failed.json.

Usage:
    python extract_infographic_by_id.py --id 123
    python extract_infographic_by_id.py --id 123 --infographic-dir /path/to/infographic --output-dir /path/to/output
"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional


def get_infographic_file_path(image_id: int, infographic_dir: str, offset: int = 0) -> str:
    """
    Get the path to the infographic JSON file containing data for the given image ID.
    - Infographic ID starts from 1 (image_id)
    - Files are chunked into groups of 50
    - File naming: infographic{file_index:06d}.json
    
    Args:
        image_id: The infographic ID to search for
        infographic_dir: Directory containing infographic files
        offset: ID offset to adjust file calculation
               Example: If ID 19030 should be at position 1 in file 000001, use offset=19029
                        If ID 19080 should be at position 51 in file 000002, use offset=19029
    """
    # Adjust ID by offset (offset shifts the ID range)
    # If offset=19029: ID 19030 becomes position 1, ID 19080 becomes position 51
    adjusted_id = image_id - offset
    
    # Ensure adjusted_id is at least 1
    if adjusted_id < 1:
        adjusted_id = 1
    
    # Calculate which file chunk it belongs to
    # Position 1-50 → file 1, Position 51-100 → file 2, etc.
    file_index = (adjusted_id - 1) // 50 + 1
    filename = f"infographic{file_index:06d}.json"
    return os.path.join(infographic_dir, filename)


def load_json(filepath: str) -> Any:
    """Load JSON file safely."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def find_infographic_entry_by_id(infographic_data: List[Dict], target_id: int) -> Optional[Dict]:
    """Find infographic entry with matching infographic_id."""
    for entry in infographic_data:
        if entry.get('infographic_id') == target_id:
            return entry
    return None


def extract_and_save_infographic(
    infographic_id: int,
    infographic_dir: str,
    output_dir: str,
    offset: int = 0
) -> bool:
    """
    Extract infographic entry by ID and save to failed.json.
    
    Args:
        infographic_id: The infographic ID to extract
        infographic_dir: Directory containing infographic JSON files
        output_dir: Output directory for failed.json
        offset: ID offset to adjust file calculation
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Searching for infographic ID: {infographic_id}")
    if offset > 0:
        print(f"Using offset: {offset} (adjusted ID: {infographic_id - offset})")
    
    # Get the file path containing this ID
    infographic_file_path = get_infographic_file_path(infographic_id, infographic_dir, offset)
    
    if not os.path.exists(infographic_file_path):
        print(f"Error: Infographic file not found: {infographic_file_path}")
        return False
    
    print(f"Loading data from: {infographic_file_path}")
    
    # Load infographic data
    infographic_data = load_json(infographic_file_path)
    if not infographic_data:
        print(f"Error: Could not load infographic data from {infographic_file_path}")
        return False
    
    # Find the specific infographic entry
    infographic_entry = find_infographic_entry_by_id(infographic_data, infographic_id)
    if not infographic_entry:
        print(f"Error: No infographic entry found for ID {infographic_id} in {infographic_file_path}")
        return False
    
    print(f"Found infographic entry for ID: {infographic_id}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to failed.json as a list (following ocr_filter.py format)
    failed_json_path = os.path.join(output_dir, 'failed.json')
    
    # Check if failed.json already exists
    existing_entries = []
    if os.path.exists(failed_json_path):
        existing_data = load_json(failed_json_path)
        if existing_data and isinstance(existing_data, list):
            existing_entries = existing_data
            # Check if this ID already exists
            existing_ids = [entry.get('infographic_id') for entry in existing_entries]
            if infographic_id in existing_ids:
                print(f"Warning: Infographic ID {infographic_id} already exists in failed.json")
                response = input("Do you want to replace it? (y/n): ").lower()
                if response != 'y':
                    print("Operation cancelled.")
                    return False
                # Remove existing entry with same ID
                existing_entries = [e for e in existing_entries if e.get('infographic_id') != infographic_id]
    
    # Add new entry
    existing_entries.append(infographic_entry)
    
    # Save to file
    with open(failed_json_path, 'w', encoding='utf-8') as f:
        json.dump(existing_entries, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Successfully saved infographic entry to: {failed_json_path}")
    print(f"  Total entries in failed.json: {len(existing_entries)}")
    
    # Display entry details
    print("\nEntry details:")
    print(f"  Infographic ID: {infographic_entry.get('infographic_id')}")
    print(f"  QID: {infographic_entry.get('qid', 'N/A')}")
    if 'generated_infographic' in infographic_entry:
        preview = infographic_entry['generated_infographic'][:100]
        print(f"  Generated text: {preview}...")
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Extract infographic entry by ID and save to failed.json'
    )
    parser.add_argument(
        '--id',
        type=int,
        required=True,
        help='Infographic ID to extract'
    )
    parser.add_argument(
        '--infographic-dir',
        type=str,
        default='/home/thinhnp/hf_vqa/src/data/narrator/infographic_val',
        help='Directory containing infographic JSON files (default: /home/thinhnp/hf_vqa/src/data/narrator/infographic_val)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/thinhnp/hf_vqa/src/data/narrator/infographic_val',
        help='Output directory for failed.json (default: /home/thinhnp/hf_vqa/src/data/narrator/infographic_val)'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing failed.json without prompting (default: prompt if ID exists)'
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        help='ID offset for file calculation. Example: If ID 19030 is at position 1 in file 000001, use --offset 19029. Default: 0'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.id < 1:
        print("Error: Infographic ID must be >= 1")
        return
    
    if not os.path.exists(args.infographic_dir):
        print(f"Error: Infographic directory not found: {args.infographic_dir}")
        return
    
    print("=== Extract Infographic by ID ===")
    print(f"Infographic directory: {args.infographic_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.offset > 0:
        print(f"ID offset: {args.offset}")
    print()
    
    # Extract and save
    success = extract_and_save_infographic(
        args.id,
        args.infographic_dir,
        args.output_dir,
        args.offset
    )
    
    if success:
        print("\n✓ Operation completed successfully!")
    else:
        print("\n✗ Operation failed!")
        exit(1)


if __name__ == '__main__':
    main()
