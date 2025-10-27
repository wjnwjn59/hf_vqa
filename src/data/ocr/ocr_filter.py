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


def get_wiki_file_path(image_id: int, bizgen_format_dir: str) -> str:
    """
    Get the path to the wiki JSON file containing data for the given image ID.
    Based on merge_infographic_bboxes.py logic:
    - Wiki ID starts from 1 (image_id)
    - Files are chunked into groups of 50
    - File naming: wiki{file_index:06d}.json
    """
    wiki_id = image_id  # Image ID directly maps to wiki ID
    file_index = (wiki_id - 1) // 50 + 1  # Calculate which file chunk it belongs to
    filename = f"wiki{file_index:06d}.json"
    return os.path.join(bizgen_format_dir, filename)


def load_json(filepath: str) -> Any:
    """Load JSON file safely."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def find_wiki_entry_by_id(wiki_data: List[Dict], target_id: int) -> Optional[Dict]:
    """Find wiki entry with matching index (wiki ID)."""
    for entry in wiki_data:
        if entry.get('index') == target_id:
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


def process_single_image(
    image_path: str, 
    image_id: int, 
    bizgen_format_dir: str,
    ocr_model: PaddleOCR,
    threshold: float = 0.5
) -> Optional[Dict]:
    """
    Process a single image: OCR + compare with JSON data.
    
    Returns:
        Dict with analysis results if image should be filtered, None otherwise
    """
    print(f"Processing image {image_id} ({os.path.basename(image_path)})...")
    
    # Get corresponding wiki file
    wiki_file_path = get_wiki_file_path(image_id, bizgen_format_dir)
    
    if not os.path.exists(wiki_file_path):
        print(f"  Warning: Wiki file not found: {wiki_file_path}")
        return None
    
    # Load wiki data
    wiki_data = load_json(wiki_file_path)
    if not wiki_data:
        print(f"  Error: Could not load wiki data from {wiki_file_path}")
        return None
    
    # Find the specific wiki entry
    wiki_entry = find_wiki_entry_by_id(wiki_data, image_id)
    if not wiki_entry:
        print(f"  Warning: No wiki entry found for image ID {image_id}")
        return None
    
    # Extract text content directly from text layers
    json_texts = []
    
    if 'layers_all' in wiki_entry:
        for layer in wiki_entry['layers_all']:
            if layer.get('category') == 'text':
                # Try to get text from 'text' field first, then from 'caption'
                text_content = layer.get('text', '')
                if not text_content and 'caption' in layer:
                    # Extract text from caption like 'Text "content" in <color-x>, <font-y>'
                    caption = layer['caption']
                    # Find text between quotes
                    match = re.search(r'Text "([^"]*)"', caption)
                    if match:
                        text_content = match.group(1)
                
                if text_content.strip():
                    json_texts.append(text_content.strip())
    
    # Perform OCR
    ocr_texts, ocr_word_count = perform_ocr(image_path, ocr_model)
    
    # Calculate similarity ratio
    similarity_ratio = calculate_text_similarity_ratio(json_texts, ocr_texts)
    
    print(f"  Text similarity ratio (Jaccard): {similarity_ratio:.3f}")
    
    # Determine if image should be filtered
    # Filter based on Jaccard similarity - if similarity is below threshold, filter the image
    should_filter = similarity_ratio < threshold
    
    print(f"  Should filter: {should_filter} (Jaccard similarity {similarity_ratio:.3f} < threshold {threshold:.3f})")
    
    if should_filter:
        # Rename the image file to add a _faults suffix before the extension
        try:
            image_dir = os.path.dirname(image_path)
            image_base = os.path.basename(image_path)
            name, ext = os.path.splitext(image_base)
            # Avoid double-appending if already suffixed
            if not name.endswith('_faults'):
                new_name = f"{name}_faults{ext}"
                new_path = os.path.join(image_dir, new_name)
                # If target exists, append a counter
                counter = 1
                while os.path.exists(new_path):
                    new_name = f"{name}_faults_{counter}{ext}"
                    new_path = os.path.join(image_dir, new_name)
                    counter += 1
                os.rename(image_path, new_path)
                print(f"  Renamed image: {image_base} -> {os.path.basename(new_path)}")
                # Update local variables so returned paths point to the new file
                image_path = new_path
                image_base = os.path.basename(new_path)
        except Exception as e:
            print(f"  Warning: failed to rename image {image_path}: {e}")

        return {
            'image_id': image_id,
            'image_filename': os.path.basename(image_path),
            'image_path': image_path,
            'wiki_file': os.path.basename(wiki_file_path),
            'text_similarity_ratio': similarity_ratio,
            'ocr_percentage': round(similarity_ratio * 100.0, 2),
            'json_texts': json_texts,
            'ocr_texts': ocr_texts,
            'reason': f'Jaccard similarity ({similarity_ratio:.3f}) between JSON and OCR texts is below threshold ({threshold:.3f})'
        }
    
    return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='OCR filter script to find images with low Jaccard text similarity between JSON and OCR'
    )
    parser.add_argument(
        '--start-id',
        type=int,
        default=1,
        help='Start image ID (default: 1)'
    )
    parser.add_argument(
        '--end-id',
        type=int,
        default=None,
        help='End image ID (default: auto-detect from images)'
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
        default='src/data/create_data/bizgen/output/infographic_data_no_parse',
        help='Directory containing images'
    )
    parser.add_argument(
        '--bizgen-dir',
        type=str,
        default='src/data/create_data/output/bizgen_format',
        help='Directory containing bizgen_format JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='src/data/create_data/output/ocr_filter',
        help='Output directory for filtered results'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    if args.end_id is None:
        args.end_id = max(img_id for img_id, _ in image_files) if image_files else args.start_id
    
    image_files = [(img_id, full_path) for img_id, full_path in image_files 
                   if args.start_id <= img_id <= args.end_id]
    
    print(f"Found {len(image_files)} images to process (ID range: {args.start_id}-{args.end_id})")
    if len(image_files) > 0:
        print("Sample images found:")
        for img_id, img_path in image_files[:5]:  # Show first 5 as examples
            print(f"  ID {img_id}: {os.path.relpath(img_path, args.images_dir)}")
    print(f"Using Jaccard similarity threshold: {args.threshold:.3f}")
    print(f"Images directory: {args.images_dir}")
    print(f"Bizgen directory: {args.bizgen_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Initialize OCR model once (expensive operation)
    print("Initializing PaddleOCR model...")
    ocr_model = PaddleOCR(use_textline_orientation=True, lang='en')
    print("OCR model initialized successfully!")
    print()
    
    # Process each image
    filtered_results = []
    processed_count = 0
    
    for image_id, full_path in image_files:
        image_path = full_path  # full_path already contains the complete path
        
        result = process_single_image(
            image_path, 
            image_id, 
            args.bizgen_dir, 
            ocr_model,
            args.threshold
        )
        
        if result:
            filtered_results.append(result)
            print(f"  ✓ Image {image_id} filtered")
        else:
            print(f"  - Image {image_id} passed")
        
        processed_count += 1
        print()
    
    # Save results
    output_file = os.path.join(args.output_dir, 'filtered_images.json')
    
    summary = {
        'metadata': {
            'total_images_processed': processed_count,
            'total_images_filtered': len(filtered_results),
            'filter_threshold': args.threshold,
            'start_id': args.start_id,
            'end_id': args.end_id,
            'images_directory': os.path.abspath(args.images_dir),
            'bizgen_directory': os.path.abspath(args.bizgen_dir),
            'filter_ratio': len(filtered_results) / processed_count if processed_count > 0 else 0.0
        },
        'filtered_images': filtered_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("=== SUMMARY ===")
    print(f"Total images processed: {processed_count}")
    print(f"Images filtered: {len(filtered_results)}")
    print(f"Filter rate: {len(filtered_results)/processed_count:.1%}" if processed_count > 0 else "0%")
    print(f"Results saved to: {output_file}")
    
    if filtered_results:
        print("\nFiltered images:")
        for result in filtered_results:
            print(f"  - {result['image_filename']} (ID: {result['image_id']}) - {result['reason']}")


if __name__ == '__main__':
    main()