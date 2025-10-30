#!/usr/bin/env python3
"""
Script to merge ChatGPT generated outputs into the format compatible with generate_stage_infographic.py

This script processes ChatGPT outputs from individual JSON files (sentences, summaries, figures, final_description, combined)
and converts them into the expected format for the infographic generation pipeline.
"""

import json
import os
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path

def load_json_file(filepath: str) -> Optional[Dict]:
    """Load JSON file and return parsed content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {filepath}: {e}")
        return None

def merge_figures_with_summaries(summaries: List[Dict], figures: List[Dict]) -> List[Dict]:
    """Merge figures with summaries by ID to create the expected merged format"""
    merged_items = []
    
    # Create a mapping of summaries by ID
    summary_map = {item["id"]: item for item in summaries}
    
    # Create a mapping of figures by ID
    figure_map = {item["id"]: item for item in figures}
    
    # Get all unique IDs
    all_ids = set(summary_map.keys()) | set(figure_map.keys())
    
    for item_id in sorted(all_ids):
        merged_item = {
            "id": item_id,
            "summary": summary_map.get(item_id, {}).get("summary", ""),
            "ideas": figure_map.get(item_id, {}).get("ideas", [])
        }
        merged_items.append(merged_item)
    
    return merged_items

def process_chatgpt_output_directory(input_dir: str, sample_id: int = 1) -> Optional[Dict[str, Any]]:
    """
    Process a directory containing ChatGPT outputs and convert to the expected format
    
    Args:
        input_dir: Directory containing the ChatGPT output files
        sample_id: ID to assign to this sample
    
    Returns:
        Dictionary in the format expected by generate_stage_infographic.py
    """
    
    # File paths
    sentences_file = os.path.join(input_dir, "sentences.json")
    summaries_file = os.path.join(input_dir, "summaries.json")
    figures_file = os.path.join(input_dir, "figures.json")
    final_desc_file = os.path.join(input_dir, "final_description.json")
    combined_file = os.path.join(input_dir, "combined.json")
    
    # Try to load from individual files first
    sentences_data = load_json_file(sentences_file)
    summaries_data = load_json_file(summaries_file)
    figures_data = load_json_file(figures_file)
    final_desc_data = load_json_file(final_desc_file)
    
    # If individual files don't exist, try combined file
    if not all([sentences_data, summaries_data, figures_data, final_desc_data]):
        print(f"Some individual files missing, trying combined.json...")
        combined_data = load_json_file(combined_file)
        if combined_data:
            sentences_data = {"sentences": combined_data.get("sentences", [])}
            summaries_data = {"summaries": combined_data.get("summaries", [])}
            figures_data = {"figures": combined_data.get("figures", [])}
            final_desc_data = {"final_description": combined_data.get("final_description", "")}
    
    # Validate that we have all required data
    if not all([sentences_data, summaries_data, figures_data, final_desc_data]):
        print(f"Error: Missing required data in {input_dir}")
        return None
    
    # Extract data
    sentences = sentences_data.get("sentences", [])
    summaries = summaries_data.get("summaries", [])
    figures = figures_data.get("figures", [])
    final_description = final_desc_data.get("final_description", "")
    
    # Validate data structure
    if not sentences or not summaries or not figures or not final_description:
        print(f"Error: Empty data found in {input_dir}")
        return None
    
    # Merge figures with summaries
    merged_figures = merge_figures_with_summaries(summaries, figures)
    
    # Create the output format compatible with generate_stage_infographic.py
    result = {
        "id": f"chatgpt_sample_{sample_id:06d}",
        "title": f"ChatGPT Generated Sample {sample_id}",
        "context": " ".join([sent["text"] for sent in sentences]),  # Reconstruct context from sentences
        "qa_pairs": [],  # ChatGPT outputs don't have QA pairs
        "keywords": [],  # No keywords from ChatGPT outputs
        "keywords_found": [],  # No keyword checking needed
        "has_answerable_questions": False,  # No QA pairs
        "skipped_keyword_check": True,  # Skip keyword check since no QA pairs
        "retry_count": 0,
        "final_seed": None,
        "generated_infographic": {
            "sentences": sentences,
            "summaries": summaries,
            "figures": merged_figures,
            "full_image_caption": final_description
        },
        "success": True,
        "infographic_id": sample_id,
        "source": "chatgpt_output"
    }
    
    return result

def process_multiple_directories(input_base_dir: str, output_file: str, max_samples: Optional[int] = None):
    """
    Process multiple ChatGPT output directories and save them to a single JSON file
    
    Args:
        input_base_dir: Base directory containing sample directories
        output_file: Output JSON file path
        max_samples: Maximum number of samples to process (None for all)
    """
    
    if not os.path.exists(input_base_dir):
        raise FileNotFoundError(f"Input directory not found: {input_base_dir}")
    
    # Find all sample directories
    sample_dirs = []
    for item in os.listdir(input_base_dir):
        item_path = os.path.join(input_base_dir, item)
        if os.path.isdir(item_path) and item.startswith("sample_"):
            sample_dirs.append(item_path)
    
    sample_dirs.sort()  # Sort for consistent ordering
    
    if max_samples:
        sample_dirs = sample_dirs[:max_samples]
    
    print(f"Found {len(sample_dirs)} sample directories")
    
    results = []
    failed_count = 0
    
    for i, sample_dir in enumerate(sample_dirs, 1):
        print(f"Processing {sample_dir}...")
        
        result = process_chatgpt_output_directory(sample_dir, i)
        if result:
            results.append(result)
            print(f"  ✓ Successfully processed sample {i}")
        else:
            failed_count += 1
            print(f"  ✗ Failed to process sample {i}")
    
    if results:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Successfully saved {len(results)} samples to {output_file}")
        if failed_count > 0:
            print(f"✗ Failed to process {failed_count} samples")
    else:
        print("✗ No valid samples found to save")

def main():
    parser = argparse.ArgumentParser(description='Merge ChatGPT outputs into generate_stage_infographic.py format')
    parser.add_argument('--input_dir', type=str, 
                        help='Directory containing ChatGPT output directories (e.g., /mnt/VLAI_data/output/outputs/)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (default: all)')
    parser.add_argument('--single_dir', type=str, default=None,
                        help='Process a single directory instead of multiple directories')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.single_dir and not args.input_dir:
        parser.error("Either --input_dir or --single_dir must be provided")
    
    print("="*60)
    print("ChatGPT Output Merger")
    print("="*60)
    
    if args.single_dir:
        # Process single directory
        print(f"Processing single directory: {args.single_dir}")
        
        result = process_chatgpt_output_directory(args.single_dir, 1)
        if result:
            # Ensure output directory exists
            output_dir = os.path.dirname(args.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save as list with single item for consistency
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump([result], f, ensure_ascii=False, indent=2)
            
            print(f"✓ Successfully saved single sample to {args.output_file}")
        else:
            print("✗ Failed to process the directory")
    else:
        # Process multiple directories
        print(f"Processing multiple directories from: {args.input_dir}")
        print(f"Output file: {args.output_file}")
        if args.max_samples:
            print(f"Max samples: {args.max_samples}")
        
        process_multiple_directories(args.input_dir, args.output_file, args.max_samples)
    
    print("\n" + "="*60)
    print("Merge Complete!")
    print("="*60)

if __name__ == "__main__":
    main()