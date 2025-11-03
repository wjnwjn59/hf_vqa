import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
import glob


def load_training_files(input_dir: str) -> List[Dict[str, Any]]:
    """Load all training data files from the input directory."""
    pattern = os.path.join(input_dir, "narrator_*_training.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No training files found matching pattern: {pattern}")
        return []
    
    print(f"Found {len(files)} training data files:")
    for file in sorted(files):
        print(f"  - {file}")
    
    all_data = []
    for file in sorted(files):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  Loaded {len(data)} samples from {os.path.basename(file)}")
                all_data.extend(data)
        except Exception as e:
            print(f"  ✗ Error loading {file}: {e}")
    
    return all_data


def save_training_data(data: List[Dict[str, Any]], output_file: str) -> None:
    """Save training data in both JSON and JSONL formats."""
    # Save JSON format
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Save JSONL format
    jsonl_file = output_file.replace('.json', '.jsonl')
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved merged training data:")
    print(f"  JSON file : {output_file}")
    print(f"  JSONL file: {jsonl_file}")


def print_statistics(data: List[Dict[str, Any]]) -> None:
    """Print statistics about the merged training dataset."""
    if not data:
        print("No data to analyze")
        return
    
    total_samples = len(data)
    total_conversations = sum(len(item.get('conversations', [])) for item in data)
    total_qa_pairs = total_conversations // 2  # Each QA pair has 2 conversation turns
    
    # Analyze conversation lengths
    conv_lengths = [len(item.get('conversations', [])) // 2 for item in data]
    avg_qa_pairs = sum(conv_lengths) / len(conv_lengths) if conv_lengths else 0
    min_qa_pairs = min(conv_lengths) if conv_lengths else 0
    max_qa_pairs = max(conv_lengths) if conv_lengths else 0
    
    # Count unique images
    images = set(item.get('image', '') for item in data)
    unique_images = len(images)
    
    print("\n" + "="*60)
    print("MERGED TRAINING DATASET STATISTICS")
    print("="*60)
    print(f"Total training samples    : {total_samples:,}")
    print(f"Total conversation turns  : {total_conversations:,}")
    print(f"Total QA pairs           : {total_qa_pairs:,}")
    print(f"Unique images            : {unique_images:,}")
    print(f"Average QA pairs per sample: {avg_qa_pairs:.2f}")
    print(f"Min QA pairs per sample  : {min_qa_pairs}")
    print(f"Max QA pairs per sample  : {max_qa_pairs}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Merge training data files from parallel narrator pipeline')
    parser.add_argument('--input-dir', type=str, default='training_data',
                        help='Directory containing training data files to merge')
    parser.add_argument('--output-file', type=str, default='training_data/merged_training.json',
                        help='Output file for merged training data')
    parser.add_argument('--clean-input', action='store_true',
                        help='Remove individual training files after merging')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("MERGING TRAINING DATA")
    print("="*60)
    print(f"Input directory : {args.input_dir}")
    print(f"Output file     : {args.output_file}")
    print()
    
    # Load all training files
    all_data = load_training_files(args.input_dir)
    
    if not all_data:
        print("No training data found to merge!")
        return
    
    print(f"\nTotal samples loaded: {len(all_data)}")
    
    # Remove duplicates based on ID
    seen_ids = set()
    unique_data = []
    duplicates = 0
    
    for item in all_data:
        item_id = item.get('id', '')
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            unique_data.append(item)
        else:
            duplicates += 1
    
    if duplicates > 0:
        print(f"Removed {duplicates} duplicate samples")
        print(f"Unique samples: {len(unique_data)}")
    
    # Save merged data
    save_training_data(unique_data, args.output_file)
    
    # Print statistics
    print_statistics(unique_data)
    
    # Clean up input files if requested
    if args.clean_input:
        pattern = os.path.join(args.input_dir, "narrator_*_training.json*")
        files_to_remove = glob.glob(pattern)
        for file in files_to_remove:
            try:
                os.remove(file)
                print(f"Removed: {file}")
            except Exception as e:
                print(f"Failed to remove {file}: {e}")
    
    print("\n✓ Training data merge completed successfully!")


if __name__ == "__main__":
    main()