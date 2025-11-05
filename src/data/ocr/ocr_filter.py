import os
import json
import argparse
from typing import Dict, List, Any, Tuple, Optional
import re
from paddleocr import PaddleOCR
import cv2
import numpy as np


def get_image_id_from_filename(filename: str) -> Optional[int]:
    """Extract image ID from filename like '1.png' -> 1"""
    match = re.match(r'(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return None


def get_infographic_file_path(image_id: int, infographic_dir: str) -> str:
    """
    Get the path to the infographic JSON file containing data for the given image ID.
    - Infographic ID starts from 1 (image_id)
    - Files are chunked into groups of 50
    - File naming: infographic{file_index:06d}.json
    """
    infographic_id = image_id  # Image ID directly maps to infographic ID
    file_index = (infographic_id - 1) // 50 + 1  # Calculate which file chunk it belongs to
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


def perform_ocr(image_path: str, ocr_model: PaddleOCR) -> Tuple[List[str], int]:
    """
    Perform OCR on image and return extracted text and word count.
    
    Args:
        image_path: Path to image file
        ocr_model: Pre-initialized PaddleOCR model
    
    Returns:
        Tuple of (list of text lines, total word count)
    """
    try:
        # Perform OCR using the pre-initialized model
        result = ocr_model.predict(image_path)
        
        if not result:
            return [], 0
        
        # Extract text from new PaddleOCR format
        text_lines = []
        total_words = 0
        
        if isinstance(result, list) and len(result) > 0:
            first_result = result[0]
            if isinstance(first_result, dict):
                # New format has 'rec_texts' and 'rec_scores' arrays
                rec_texts = first_result.get('rec_texts', [])
                rec_scores = first_result.get('rec_scores', [])
                
                for text, score in zip(rec_texts, rec_scores):
                    if score > 0.5 and text.strip():
                        text_lines.append(text.strip())
                        words = text.strip().split()
                        total_words += len(words)
        
        return text_lines, total_words
        
    except Exception as e:
        print(f"Error performing OCR on {image_path}: {e}")
        return [], 0


def calculate_text_similarity_ratio(json_texts: List[str], ocr_texts: List[str]) -> float:
    """
    Calculate the ratio of text content similarity between JSON and OCR.
    This is a simple word-based comparison.
    
    Returns:
        Ratio (0.0 to 1.0) where 1.0 means perfect match
    """
    if not ocr_texts:
        return 0.0
    
    # Convert to word sets for comparison
    json_words = set()
    for text in json_texts:
        json_words.update(text.lower().split())
    
    ocr_words = set()
    for text in ocr_texts:
        ocr_words.update(text.lower().split())
    
    if not ocr_words:
        return 0.0
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = len(json_words.intersection(ocr_words))
    union = len(json_words.union(ocr_words))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def contains_suspicious_patterns(ocr_texts: List[str]) -> Tuple[bool, str]:
    """
    Check if OCR texts contain suspicious patterns like URLs (www) or repeated characters.
    
    Args:
        ocr_texts: List of OCR text strings
    
    Returns:
        Tuple of (has_suspicious_pattern, pattern_description)
    """
    for text in ocr_texts:
        text_lower = text.lower()
        
        # Check for 4 consecutive 'w' characters (like www or wwww)
        if 'wwww' in text_lower:
            return True, f"Detected 4+ consecutive 'w' characters (URL pattern): '{text[:50]}...'"
        
        # Optional: Check for common URL patterns
        if re.search(r'www\.[a-z]+', text_lower):
            return True, f"Detected URL pattern (www.xxx): '{text[:50]}...'"
        
        # Optional: Check for excessive character repetition (like 'aaaaa', 'bbbbb')
        # This catches other OCR errors with repeated characters
        if re.search(r'([a-z])\1{4,}', text_lower):
            return True, f"Detected 5+ repeated characters: '{text[:50]}...'"
    
    return False, ""


def process_single_image(
    image_path: str, 
    image_id: int, 
    infographic_dir: str,
    ocr_model: PaddleOCR,
    threshold: float = 0.5
) -> Optional[Dict]:
    """
    Process a single image: OCR + compare with JSON data.
    
    Returns:
        Dict with analysis results if image should be filtered, None otherwise
    """
    print(f"Processing image {image_id} ({os.path.basename(image_path)})...")
    
    # Get corresponding infographic file
    infographic_file_path = get_infographic_file_path(image_id, infographic_dir)
    
    if not os.path.exists(infographic_file_path):
        print(f"  Warning: Infographic file not found: {infographic_file_path}")
        return None
    
    # Load infographic data
    infographic_data = load_json(infographic_file_path)
    if not infographic_data:
        print(f"  Error: Could not load infographic data from {infographic_file_path}")
        return None
    
    # Find the specific infographic entry
    infographic_entry = find_infographic_entry_by_id(infographic_data, image_id)
    if not infographic_entry:
        print(f"  Warning: No infographic entry found for image ID {image_id}")
        return None
    
    # Extract text content from generated_infographic field
    json_texts = []
    
    if 'generated_infographic' in infographic_entry:
        generated_text = infographic_entry['generated_infographic']
        # Extract text content from patterns like (text) markers
        # Find all text marked as (text)
        text_matches = re.findall(r'"([^"]+)"\s*\(text\)', generated_text)
        json_texts.extend([text.strip() for text in text_matches if text.strip()])
    
    # Perform OCR
    ocr_texts, ocr_word_count = perform_ocr(image_path, ocr_model)
    
    # Check for suspicious patterns (www, repeated characters, etc.)
    has_suspicious, suspicious_reason = contains_suspicious_patterns(ocr_texts)
    
    if has_suspicious:
        return {
            'infographic_entry': infographic_entry,  # Complete infographic format entry
            'image_id': image_id,
            'image_filename': os.path.basename(image_path),
            'similarity_ratio': 0.0,  # Mark as 0 for pattern-based failure
            'reason': f'Suspicious pattern: {suspicious_reason}'
        }
    
    # Calculate similarity ratio
    similarity_ratio = calculate_text_similarity_ratio(json_texts, ocr_texts)
    
    print(f"  Text similarity ratio (Jaccard): {similarity_ratio:.3f}")
    
    # Determine if image should be filtered
    # Filter based on Jaccard similarity - if similarity is below threshold, filter the image
    should_filter = similarity_ratio < threshold
    
    print(f"  Should filter: {should_filter} (Jaccard similarity {similarity_ratio:.3f} < threshold {threshold:.3f})")
    
    if should_filter:
        # Return the infographic entry for failed images
        print(f"  ✗ Image {image_id} failed OCR check - will be added to failed.json for regeneration")
        return {
            'infographic_entry': infographic_entry,  # Complete infographic format entry
            'image_id': image_id,
            'image_filename': os.path.basename(image_path),
            'similarity_ratio': similarity_ratio,
            'reason': f'Jaccard similarity ({similarity_ratio:.3f}) < threshold ({threshold:.3f})'
        }
    
    return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='OCR filter script to find images with low Jaccard text similarity between JSON and OCR'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='Start file index (1-based). --start 1 processes images 1-50 from infographic000001.json'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End file index (exclusive, 1-based). --start 1 --end 3 processes 2 files (infographic000001, infographic000002) = images 1-100'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Jaccard similarity threshold for filtering (default: 0.5). Images with similarity below this value will be filtered.'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        default='src/data/bizgen/output/squad_v2',
        help='Directory containing images'
    )
    parser.add_argument(
        '--infographic-dir',
        type=str,
        default='src/data/narrator/infographic',
        help='Directory containing infographic JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='src/data/narrator/infographic',
        help='Output directory for filtered results (failed.json)'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Calculate image ID range based on file indices
    if args.end is not None:
        num_files = args.end - args.start
        start_image_id = (args.start - 1) * 50 + 1
        end_image_id = (args.end - 1) * 50  # Exclusive end
    else:
        # If no end specified, process from start file onwards (auto-detect max)
        start_image_id = (args.start - 1) * 50 + 1
        end_image_id = None  # Will be determined from available images
    
    # Get list of image files (search recursively in subdirectories)
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return
    
    image_files = []
    # Walk through all subdirectories to find images
    for root, dirs, files in os.walk(args.images_dir):
        for filename in files:
            if filename.endswith('.png') and not filename.endswith('_bbox.png') and not filename.endswith('_lcfg.png'):
                image_id = get_image_id_from_filename(filename)
                if image_id is not None:
                    # Store full path relative to the images_dir for processing
                    full_path = os.path.join(root, filename)
                    image_files.append((image_id, full_path))
    
    # Sort by image ID
    image_files.sort(key=lambda x: x[0])
    
    # Filter by ID range
    if end_image_id is None:
        # Auto-detect max image ID
        end_image_id = max(img_id for img_id, _ in image_files) if image_files else start_image_id
    
    image_files = [(img_id, full_path) for img_id, full_path in image_files 
                   if start_image_id <= img_id <= end_image_id]
    
    # Calculate file range for display
    if args.end is not None:
        first_file = args.start
        last_file = args.end - 1
        num_files = args.end - args.start
        print(f"File range: {first_file} to {last_file} ({num_files} files)")
        print(f"Image ID range: {start_image_id} to {end_image_id}")
    else:
        print(f"File range: {args.start} onwards")
        print(f"Image ID range: {start_image_id} to {end_image_id}")
    
    print(f"Found {len(image_files)} images to process")
    if len(image_files) > 0:
        print("Sample images found:")
        for img_id, img_path in image_files[:5]:  # Show first 5 as examples
            print(f"  ID {img_id}: {os.path.relpath(img_path, args.images_dir)}")
    print(f"Using Jaccard similarity threshold: {args.threshold:.3f}")
    print(f"Images directory: {args.images_dir}")
    print(f"Infographic directory: {args.infographic_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Initialize OCR model once (expensive operation)
    print("Initializing PaddleOCR model...")
    ocr_model = PaddleOCR(use_textline_orientation=True, lang='en')
    print("OCR model initialized successfully!")
    print()
    
    # Process each image
    filtered_results = []
    failed_infographic_entries = []  # Collect infographic format entries for failed images
    processed_count = 0
    
    for image_id, full_path in image_files:
        image_path = full_path  # full_path already contains the complete path
        
        result = process_single_image(
            image_path, 
            image_id, 
            args.infographic_dir, 
            ocr_model,
            args.threshold
        )
        
        if result:
            filtered_results.append(result)
            # Add infographic entry to failed list for regeneration
            failed_infographic_entries.append(result['infographic_entry'])
            print(f"  ✓ Image {image_id} filtered (will regenerate)")
        else:
            print(f"  - Image {image_id} passed")
        
        processed_count += 1
        print()
    
    # Save failed.json with infographic format (list of infographic entries)
    failed_json_path = os.path.join(args.output_dir, 'failed.json')
    with open(failed_json_path, 'w', encoding='utf-8') as f:
        json.dump(failed_infographic_entries, f, indent=2, ensure_ascii=False)
    
    # Save summary report (metadata + failed image info)
    summary_file = os.path.join(args.output_dir, 'filtered_summary.json')
    
    summary = {
        'metadata': {
            'total_images_processed': processed_count,
            'total_images_filtered': len(filtered_results),
            'filter_threshold': args.threshold,
            'start_file': args.start,
            'end_file': args.end if args.end else 'auto',
            'start_image_id': start_image_id,
            'end_image_id': end_image_id,
            'images_directory': os.path.abspath(args.images_dir),
            'infographic_directory': os.path.abspath(args.infographic_dir),
            'filter_ratio': len(filtered_results) / processed_count if processed_count > 0 else 0.0
        },
        'filtered_images': [
            {
                'image_id': r['image_id'],
                'image_filename': r['image_filename'],
                'similarity_ratio': r['similarity_ratio'],
                'reason': r['reason']
            } for r in filtered_results
        ]
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("=== SUMMARY ===")
    print(f"Total images processed: {processed_count}")
    print(f"Images filtered: {len(filtered_results)}")
    print(f"Filter rate: {len(filtered_results)/processed_count:.1%}" if processed_count > 0 else "0%")
    print(f"\nOutput files:")
    print(f"  - Infographic format (for regeneration): {failed_json_path}")
    print(f"  - Summary report: {summary_file}")
    
    if filtered_results:
        print(f"\nFiltered images ({len(filtered_results)} total):")
        for result in filtered_results:
            print(f"  - {result['image_filename']} (ID: {result['image_id']}) - {result['reason']}")
        print(f"\n✓ Use {failed_json_path} to regenerate infographic descriptions for these images")


if __name__ == '__main__':
    main()