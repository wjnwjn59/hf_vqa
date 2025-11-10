#!/usr/bin/env python3
"""
Script to prepare NARRATOR VQA dataset in structured format.

This script reorganizes generated infographic data into:
- Centralized image folder
- Individual template files per image
- Individual QA files per image
- SQuAD v2 format train/val annotation files
"""

import argparse
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare NARRATOR VQA dataset in structured format"
    )
    parser.add_argument(
        "--wiki-dir",
        type=str,
        default="src/data/narrator/wiki",
        help="Source directory containing wiki*.json files"
    )
    parser.add_argument(
        "--image-source-dir",
        type=str,
        default="src/data/bizgen/output",
        help="Base directory with generated images"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="squad_v2",
        help="Dataset folder name in image source directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/thinhnp/hf_vqa/dataset",
        help="Target dataset directory"
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["train", "val"],
        help="Annotation type: train or val"
    )
    parser.add_argument(
        "--squad-file",
        type=str,
        default=None,
        help="Path to original SQuAD file for reference (optional)"
    )
    parser.add_argument(
        "--reasoning-file",
        type=str,
        default="src/data/narrator/generated_reasonings.jsonl",
        help="Path to generated_reasonings.jsonl file (optional)"
    )
    
    return parser.parse_args()


def read_wiki_files(wiki_dir: str) -> List[Tuple[int, List[Dict[Any, Any]]]]:
    """
    Read all wiki*.json files from directory.
    
    Args:
        wiki_dir: Directory containing wiki files
        
    Returns:
        List of tuples (file_number, entries_list)
    """
    wiki_dir_path = Path(wiki_dir)
    wiki_files = sorted(wiki_dir_path.glob("wiki*.json"))
    
    # Filter out special files
    wiki_files = [f for f in wiki_files if f.name not in ["failed.json", "filtered_summary.json"]]
    
    logger.info(f"Found {len(wiki_files)} wiki files in {wiki_dir}")
    
    all_data = []
    for wiki_file in wiki_files:
        # Extract file number from wiki000001.json -> 1
        file_num_str = wiki_file.stem.replace("wiki", "")
        try:
            file_num = int(file_num_str)
        except ValueError:
            logger.warning(f"Skipping file with invalid name format: {wiki_file.name}")
            continue
            
        logger.info(f"Reading {wiki_file.name}...")
        with open(wiki_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)
            
        if not isinstance(entries, list):
            logger.error(f"{wiki_file.name} does not contain a list of entries")
            continue
            
        logger.info(f"  Loaded {len(entries)} entries from {wiki_file.name}")
        all_data.append((file_num, entries))
    
    return all_data


def load_reasoning_data(reasoning_file: str) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    """
    Load reasoning data from JSONL file and index by (wiki_id, layout_index, squad_id).
    
    Args:
        reasoning_file: Path to generated_reasonings.jsonl file
        
    Returns:
        Dictionary indexed by (wiki_id, layout_index, squad_id)
    """
    reasoning_file_path = Path(reasoning_file)
    
    if not reasoning_file_path.exists():
        logger.warning(f"Reasoning file not found: {reasoning_file}")
        return {}
    
    logger.info(f"Loading reasoning data from {reasoning_file}")
    
    reasoning_index = {}
    line_count = 0
    
    try:
        with open(reasoning_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                if not line.strip():
                    continue
                    
                try:
                    entry = json.loads(line)
                    wiki_id = entry['wiki_id']
                    layout_index = entry['layout_index']
                    squad_id = entry['squad_id']
                    
                    key = (wiki_id, layout_index, squad_id)
                    reasoning_index[key] = entry
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing reasoning line {line_count}: {e}")
                    continue
        
        logger.info(f"Loaded {len(reasoning_index)} reasoning entries from {line_count} lines")
        
    except Exception as e:
        logger.error(f"Failed to load reasoning file: {e}")
        return {}
    
    return reasoning_index


def find_reasoning(reasoning_index: Dict, wiki_id: str, layout_index: int, squad_id: str) -> Dict[str, Any]:
    """
    Find reasoning entry for a specific QA pair.
    
    Args:
        reasoning_index: Indexed reasoning data
        wiki_id: Wiki file ID (e.g., "000001")
        layout_index: Image index
        squad_id: Question ID
        
    Returns:
        Reasoning entry or None if not found
    """
    key = (wiki_id, layout_index, squad_id)
    return reasoning_index.get(key, None)


def find_reasonings_for_image(reasoning_index: Dict, wiki_id: str, layout_index: int) -> List[Dict[str, Any]]:
    """
    Find all reasoning entries for a specific image.
    
    Args:
        reasoning_index: Indexed reasoning data
        wiki_id: Wiki file ID (e.g., "000001")
        layout_index: Image index
        
    Returns:
        List of reasoning entries for the image
    """
    results = []
    for (w_id, l_idx, squad_id), entry in reasoning_index.items():
        if w_id == wiki_id and l_idx == layout_index:
            results.append(entry)
    return results


def get_image_source_path(index: int, file_num: int, image_source_dir: str, dataset_name: str) -> Path:
    """
    Get the source path for an image based on its index.
    Searches for the image in all narrator* subdirectories.
    
    Args:
        index: Image index (e.g., 19030 for validation data)
        file_num: Wiki file number (not used for searching, kept for compatibility)
        image_source_dir: Base image directory
        dataset_name: Dataset folder name
        
    Returns:
        Path to source image, or None if not found
    """
    image_name = f"{index}.png"
    dataset_path = Path(image_source_dir) / dataset_name
    
    # Search for the image in all narrator* subdirectories
    if dataset_path.exists():
        # Use glob to find the image across all narrator folders
        image_matches = list(dataset_path.glob(f"narrator*//{image_name}"))
        
        if image_matches:
            # Return the first match found
            return image_matches[0]
    
    # Fallback: try the old method using file_num
    narrator_folder = f"narrator{file_num:06d}"
    source_path = dataset_path / narrator_folder / image_name
    return source_path


def create_output_directories(output_dir: str) -> Dict[str, Path]:
    """
    Create output directory structure.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Dictionary of directory paths
    """
    output_path = Path(output_dir)
    
    dirs = {
        'base': output_path,
        'images': output_path / 'images',
        'templates': output_path / 'templates',
        'qas': output_path / 'qas'
    }
    
    for dir_name, dir_path in dirs.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified directory: {dir_path}")
    
    return dirs


def copy_image(source_path: Path, dest_path: Path, index: int) -> bool:
    """
    Copy image from source to destination.
    
    Args:
        source_path: Source image path
        dest_path: Destination image path
        index: Image index for logging
        
    Returns:
        True if successful, False otherwise
    """
    if not source_path.exists():
        logger.warning(f"Image not found for index {index}: {source_path}")
        return False
    
    try:
        shutil.copy2(source_path, dest_path)
        return True
    except Exception as e:
        logger.error(f"Failed to copy image {index}: {e}")
        return False


def create_template_file(entry: Dict[Any, Any], output_path: Path) -> bool:
    """
    Create template JSON file for an image.
    
    Args:
        entry: Wiki entry containing template data
        output_path: Path to output template file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        template_data = {
            "infographic_id": entry["index"],
            "full_image_caption": entry.get("full_image_caption", ""),
            "layers_all": entry.get("layers_all", []),
            "original_bbox_index": entry.get("original_bbox_index", -1)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        logger.error(f"Failed to create template file {output_path}: {e}")
        return False


def analyze_template_layout(template_path: Path) -> Dict[str, int]:
    """
    Analyze template file to count layout elements by category.
    
    Args:
        template_path: Path to template JSON file
        
    Returns:
        Dictionary with counts for each category
    """
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
        
        layers = template_data.get("layers_all", [])
        
        # Count by category
        layout_counts = {
            "base": 0,
            "element": 0,
            "text": 0,
            "total": 0
        }
        
        for layer in layers:
            category = layer.get("category", "unknown")
            if category in layout_counts:
                layout_counts[category] += 1
            layout_counts["total"] += 1
        
        return layout_counts
        
    except Exception as e:
        logger.warning(f"Failed to analyze template {template_path}: {e}")
        return {"base": 0, "element": 0, "text": 0, "total": 0}


def create_qas_file(entry: Dict[Any, Any], output_path: Path, reasoning_index: Dict, wiki_id: str, generated_reasonings: List[Dict]) -> bool:
    """
    Create QAS JSON file for an image with reasoning data.
    
    Args:
        entry: Wiki entry containing QA data
        output_path: Path to output QAS file
        reasoning_index: Indexed reasoning data
        wiki_id: Wiki file ID
        generated_reasonings: List of generated QA reasoning entries
        
    Returns:
        True if successful, False otherwise
    """
    try:
        layout_index = entry["index"]
        
        # Process original QA pairs and add reasoning
        original_qa_pairs = entry.get("original_qa_pairs", [])
        for qa in original_qa_pairs:
            qa_id = qa.get("id", "")
            reasoning_entry = find_reasoning(reasoning_index, wiki_id, layout_index, qa_id)
            if reasoning_entry:
                qa["reasoning"] = {
                    "generated_reasoning": reasoning_entry.get("generated_reasoning", {}),
                    "merged_reasoning": reasoning_entry.get("merged_reasoning", "")
                }
        
        # Create generated QA pairs from reasoning
        generated_qa_pairs = []
        for reasoning_entry in generated_reasonings:
            generated_qa_pairs.append({
                "squad_id": reasoning_entry["squad_id"],
                "question": reasoning_entry["question"],
                "answer": reasoning_entry["ground_truth_answer"],
                "reasoning": {
                    "generated_reasoning": reasoning_entry.get("generated_reasoning", {}),
                    "merged_reasoning": reasoning_entry.get("merged_reasoning", "")
                }
            })
        
        qas_data = {
            "infographic_id": layout_index,
            "context": entry.get("context", ""),
            "original_qa_pairs": original_qa_pairs,
            "generated_qa_pairs": generated_qa_pairs
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qas_data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        logger.error(f"Failed to create QAS file {output_path}: {e}")
        return False


def create_annotation_entries(entry: Dict[Any, Any], reasoning_index: Dict, wiki_id: str, generated_reasonings: List[Dict], layout_stats: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    Create lightweight metadata annotation entries that reference QAS files.
    
    Args:
        entry: Wiki entry
        reasoning_index: Indexed reasoning data
        wiki_id: Wiki file ID
        generated_reasonings: List of generated QA reasoning entries
        layout_stats: Layout statistics from template analysis
        
    Returns:
        List of annotation entries (one per QA pair, including generated)
    """
    annotations = []
    index = entry["index"]
    qa_pairs = entry.get("original_qa_pairs", [])
    
    # Add original QA pairs - just reference to qas file
    for qa_idx, qa in enumerate(qa_pairs):
        qa_id = qa.get("id", "")
        
        annotation = {
            "id": qa_id,
            "image_id": index,
            "qas_file": f"qas/{index}.json",
            "qa_type": "original",
            "qa_index": qa_idx,
            "layout_stats": layout_stats
        }
        annotations.append(annotation)
    
    # Add generated QA pairs - also reference to qas file
    for gen_idx, reasoning_entry in enumerate(generated_reasonings):
        annotation = {
            "id": reasoning_entry["squad_id"],
            "image_id": index,
            "qas_file": f"qas/{index}.json",
            "qa_type": "generated",
            "qa_index": gen_idx,
            "layout_stats": layout_stats
        }
        annotations.append(annotation)
    
    return annotations


def write_annotations(annotations: List[Dict[str, Any]], output_path: Path, annotation_type: str):
    """
    Write annotations to JSONL file.
    
    Args:
        annotations: List of annotation entries
        output_path: Path to output file
        annotation_type: Type of annotations (train/val)
    """
    logger.info(f"Writing {len(annotations)} annotations to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for annotation in annotations:
            f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
    
    logger.info(f"Successfully wrote {annotation_type} annotations")


def process_dataset(args):
    """
    Main processing function.
    
    Args:
        args: Command line arguments
    """
    logger.info("=" * 80)
    logger.info("NARRATOR VQA Dataset Preparation")
    logger.info("=" * 80)
    logger.info(f"Wiki directory: {args.wiki_dir}")
    logger.info(f"Image source: {args.image_source_dir}/{args.dataset_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Annotation type: {args.type}")
    logger.info(f"Reasoning file: {args.reasoning_file}")
    logger.info("=" * 80)
    
    # Create output directories
    dirs = create_output_directories(args.output_dir)
    
    # Load reasoning data
    reasoning_index = load_reasoning_data(args.reasoning_file)
    
    # Read all wiki files
    wiki_data = read_wiki_files(args.wiki_dir)
    
    if not wiki_data:
        logger.error("No wiki files found to process!")
        return
    
    # Statistics
    stats = {
        'total_entries': 0,
        'processed_entries': 0,
        'skipped_missing_image': 0,
        'skipped_no_qa': 0,
        'total_annotations': 0,
        'total_reasoning_loaded': len(reasoning_index),
        'generated_qa_pairs': 0,
        'original_qa_with_reasoning': 0,
        'original_qa_without_reasoning': 0,
        'layout_stats': {
            'base': 0,
            'element': 0,
            'text': 0,
            'total': 0
        }
    }
    
    all_annotations = []
    
    # Process each wiki file
    for file_num, entries in wiki_data:
        logger.info(f"\nProcessing wiki{file_num:06d}.json with {len(entries)} entries...")
        wiki_id = f"{file_num:06d}"
        
        for entry in entries:
            stats['total_entries'] += 1
            
            # Validate entry structure
            if 'index' not in entry:
                logger.warning(f"Entry missing 'index' field, skipping")
                continue
            
            index = entry['index']
            
            # Get source image path
            source_image = get_image_source_path(
                index, file_num, args.image_source_dir, args.dataset_name
            )
            dest_image = dirs['images'] / f"{index}.png"
            
            # Copy image
            if not copy_image(source_image, dest_image, index):
                stats['skipped_missing_image'] += 1
                continue
            
            # Create template file
            template_path = dirs['templates'] / f"{index}.json"
            if not create_template_file(entry, template_path):
                continue
            
            # Analyze template layout
            layout_stats = analyze_template_layout(template_path)
            
            # Accumulate layout statistics
            for key in ['base', 'element', 'text', 'total']:
                stats['layout_stats'][key] += layout_stats[key]
            
            # Find reasoning for this image
            image_reasonings = find_reasonings_for_image(reasoning_index, wiki_id, index)
            generated_reasonings = [r for r in image_reasonings if r['squad_id'].startswith('gen_')]
            
            # Create QAS file with reasoning
            qas_path = dirs['qas'] / f"{index}.json"
            if not create_qas_file(entry, qas_path, reasoning_index, wiki_id, generated_reasonings):
                continue
            
            # Count reasoning statistics
            original_qa_pairs = entry.get("original_qa_pairs", [])
            for qa in original_qa_pairs:
                qa_id = qa.get("id", "")
                if find_reasoning(reasoning_index, wiki_id, index, qa_id):
                    stats['original_qa_with_reasoning'] += 1
                else:
                    stats['original_qa_without_reasoning'] += 1
            
            stats['generated_qa_pairs'] += len(generated_reasonings)
            
            # Check if entry has QA pairs (original or generated)
            if not original_qa_pairs and not generated_reasonings:
                logger.debug(f"Entry {index} has no QA pairs, skipping annotation generation")
                stats['skipped_no_qa'] += 1
                # Still count as processed since we copied image and created template
                stats['processed_entries'] += 1
                continue
            
            # Create annotation entries with reasoning and layout stats
            annotation_entries = create_annotation_entries(entry, reasoning_index, wiki_id, generated_reasonings, layout_stats)
            all_annotations.extend(annotation_entries)
            stats['total_annotations'] += len(annotation_entries)
            
            stats['processed_entries'] += 1
            
            if stats['processed_entries'] % 100 == 0:
                logger.info(f"  Processed {stats['processed_entries']} entries...")
    
    # Write annotations
    annotation_file = dirs['base'] / f"{args.type}_annotations.jsonl"
    write_annotations(all_annotations, annotation_file, args.type)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total entries found: {stats['total_entries']}")
    logger.info(f"Successfully processed: {stats['processed_entries']}")
    logger.info(f"Skipped (missing image): {stats['skipped_missing_image']}")
    logger.info(f"Skipped (no QA pairs): {stats['skipped_no_qa']}")
    logger.info(f"\nReasoning Statistics:")
    logger.info(f"  Total reasoning entries loaded: {stats['total_reasoning_loaded']}")
    logger.info(f"  Generated QA pairs added: {stats['generated_qa_pairs']}")
    logger.info(f"  Original QA pairs with reasoning: {stats['original_qa_with_reasoning']}")
    logger.info(f"  Original QA pairs without reasoning: {stats['original_qa_without_reasoning']}")
    logger.info(f"\nLayout Statistics:")
    logger.info(f"  Total base layers: {stats['layout_stats']['base']}")
    logger.info(f"  Total element layers (images): {stats['layout_stats']['element']}")
    logger.info(f"  Total text layers: {stats['layout_stats']['text']}")
    logger.info(f"  Total layers: {stats['layout_stats']['total']}")
    if stats['processed_entries'] > 0:
        logger.info(f"  Average layers per infographic: {stats['layout_stats']['total'] / stats['processed_entries']:.2f}")
        logger.info(f"  Average element layers per infographic: {stats['layout_stats']['element'] / stats['processed_entries']:.2f}")
        logger.info(f"  Average text layers per infographic: {stats['layout_stats']['text'] / stats['processed_entries']:.2f}")
    logger.info(f"\nTotal annotation entries: {stats['total_annotations']}")
    logger.info(f"\nOutput location: {args.output_dir}")
    logger.info(f"  - Images: {dirs['images']}")
    logger.info(f"  - Templates: {dirs['templates']}")
    logger.info(f"  - QAS: {dirs['qas']}")
    logger.info(f"  - Annotations: {annotation_file}")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        process_dataset(args)
        logger.info("\nDataset preparation completed successfully!")
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()

