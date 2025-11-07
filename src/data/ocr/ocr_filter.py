import os
import json
import argparse
from typing import Dict, List, Any, Tuple, Optional
import re
from paddleocr import PaddleOCR
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker


def extract_text_elements(full_caption: str) -> List[str]:
    """
    Extract text content from caption (quoted text with (text) tag).
    Format: "text content" (text) or "text content" (text). or "text content" (text),
    Also supports: "'text content' (text)" with single quotes inside double quotes
    
    Args:
        full_caption: The full image caption text
        
    Returns:
        List of text strings
    """
    text_elements = []
    
    # Pattern 1: Match "content" (text) with optional punctuation after
    # Matches: (text), (text). (text), (text); etc.
    pattern1 = r'"([^"]+)"\s*\(text\)[.,;:!?]?'
    matches1 = re.findall(pattern1, full_caption, re.IGNORECASE)
    
    # Pattern 2: Match "'content' (text)" - single quotes inside double quotes
    # Matches: "'content' (text)", "'content' (text).", etc.
    pattern2 = r"\"'([^']+)'\s*\(text\)\s*\"[.,;:!?]?"
    matches2 = re.findall(pattern2, full_caption, re.IGNORECASE)
    
    # Combine matches from both patterns
    all_matches = matches1 + matches2
    
    for text_content in all_matches:
        text_elements.append(text_content.strip())
    
    return text_elements


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


def group_ocr_texts_to_sentences(ocr_texts: List[str]) -> List[str]:
    """
    Convert list of OCR text fragments into coherent sentences using simple heuristics.
    
    Args:
        ocr_texts: List of OCR text fragments
        
    Returns:
        List of candidate sentences
    """
    if not ocr_texts:
        return []
    
    # Concatenate all OCR texts with spaces
    full_text = ' '.join(ocr_texts)
    
    # Split on sentence-ending punctuation followed by space or end
    # Pattern: split on . ! ? followed by space or end of string
    sentences = re.split(r'[.!?](?:\s+|$)', full_text)
    
    # Filter and clean sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Filter out very short fragments (< 3 characters)
        if len(sentence) >= 3:
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


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


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity with adaptive strategy:
    - For expected text (shorter), use intersection / len(expected_words)
    - This means: what % of expected words appear in OCR text?
    
    Args:
        text1: Expected text (typically shorter)
        text2: OCR text (typically longer)
        
    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    # Convert to word sets (lowercase)
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate intersection
    intersection = len(words1.intersection(words2))
    
    # Use expected text length as denominator
    # This gives: what percentage of expected words are found in OCR?
    return intersection / len(words1)


def apply_spell_correction(words: List[str], spell_checker: SpellChecker) -> List[str]:
    """
    Apply spell correction to a list of words.
    
    Args:
        words: List of words to correct
        spell_checker: Pre-initialized SpellChecker instance
        
    Returns:
        List of corrected words
    """
    corrected_words = []
    
    for word in words:
        # Get corrected version
        corrected = spell_checker.correction(word)
        
        # If correction returns None, keep original word
        if corrected is None:
            corrected_words.append(word.lower())
        else:
            corrected_words.append(corrected.lower())
    
    return corrected_words


def calculate_word_coverage(
    expected_text: str, 
    ocr_texts: List[str],
    spell_checker: SpellChecker
) -> Tuple[float, int, int]:
    """
    Calculate word coverage: % of expected words found in spell-corrected OCR.
    
    Args:
        expected_text: Text that should appear
        ocr_texts: Raw OCR text fragments
        spell_checker: Pre-initialized SpellChecker
        
    Returns:
        Tuple of (coverage_ratio, words_found, words_expected)
    """
    # Extract expected words (lowercase, unique)
    expected_words = set(expected_text.lower().split())
    
    if not expected_words:
        return 0.0, 0, 0
    
    # Concatenate all OCR texts
    full_ocr = ' '.join(ocr_texts).lower()
    
    # Extract OCR words
    ocr_words_raw = full_ocr.split()
    
    # Apply spell correction to OCR words
    ocr_words_corrected = apply_spell_correction(ocr_words_raw, spell_checker)
    
    # Create set of corrected OCR words
    ocr_word_set = set(ocr_words_corrected)
    
    # Count how many expected words are found in corrected OCR
    words_found = len(expected_words.intersection(ocr_word_set))
    words_expected = len(expected_words)
    
    # Calculate coverage ratio
    coverage = words_found / words_expected if words_expected > 0 else 0.0
    
    return coverage, words_found, words_expected


def match_texts_with_bert(
    expected_texts: List[str], 
    ocr_texts: List[str], 
    bert_model: SentenceTransformer,
    spell_checker: SpellChecker,
    threshold: float = 0.9,
    jaccard_fallback_threshold: float = 0.6,
    word_coverage_threshold: float = 0.9
) -> Tuple[int, int, List[Dict]]:
    """
    Match expected texts with OCR texts using 3-tier approach:
    1. BERT semantic similarity (primary, threshold >= 0.9)
    2. Jaccard word overlap (fallback, threshold >= 0.6)
    3. Word-level coverage with spell correction (final fallback, threshold >= 0.9)
    
    NOTE: OCR texts are spell-corrected BEFORE all matching methods.
    
    Args:
        expected_texts: Texts that should appear in image
        ocr_texts: Raw OCR text fragments
        bert_model: Pre-loaded BERT model
        spell_checker: Pre-loaded SpellChecker
        threshold: BERT similarity threshold for matching (default: 0.9)
        jaccard_fallback_threshold: Jaccard threshold for fallback (default: 0.6)
        word_coverage_threshold: Word coverage threshold for spell-corrected matching (default: 0.9)
        
    Returns:
        Tuple of (matched_count, total_expected, match_details)
        - matched_count: Number of expected texts that found a match
        - total_expected: Total number of expected texts
        - match_details: List of dicts with match information
    """
    total_expected = len(expected_texts)
    
    # If no expected texts, consider it a pass
    if total_expected == 0:
        return 0, 0, []
    
    # Apply spell correction to each OCR text fragment while preserving structure
    ocr_texts_corrected = []
    all_corrected_words = []  # Keep all corrected words for word-level matching
    
    for text in ocr_texts:
        words = text.split()
        corrected_words = apply_spell_correction(words, spell_checker)
        # Reconstruct the text with corrected words
        ocr_texts_corrected.append(' '.join(corrected_words))
        all_corrected_words.extend(corrected_words)
    
    # Group corrected OCR texts into sentences
    ocr_sentences = group_ocr_texts_to_sentences(ocr_texts_corrected)
    
    # If no OCR sentences extracted, all expected texts fail
    if not ocr_sentences:
        match_details = [
            {
                'expected_text': text,
                'best_ocr_match': None,
                'bert_similarity': 0.0,
                'jaccard_similarity': 0.0,
                'match_method': None,
                'matched': False
            }
            for text in expected_texts
        ]
        return 0, total_expected, match_details
    
    # Encode all expected texts and OCR sentences using BERT
    expected_embeddings = bert_model.encode(expected_texts, convert_to_tensor=False)
    ocr_embeddings = bert_model.encode(ocr_sentences, convert_to_tensor=False)
    
    # Match each expected text with OCR sentences
    matched_count = 0
    match_details = []
    
    for i, expected_text in enumerate(expected_texts):
        expected_emb = expected_embeddings[i]
        
        # Calculate BERT cosine similarity with all OCR sentences
        bert_similarities = []
        for j, ocr_emb in enumerate(ocr_embeddings):
            # Cosine similarity
            similarity = np.dot(expected_emb, ocr_emb) / (
                np.linalg.norm(expected_emb) * np.linalg.norm(ocr_emb)
            )
            bert_similarities.append((similarity, ocr_sentences[j], j))
        
        # Find best BERT match
        max_bert_similarity, best_bert_match, best_idx = max(bert_similarities, key=lambda x: x[0])
        
        # Check if BERT match is above threshold
        is_matched = max_bert_similarity >= threshold
        match_method = 'BERT'
        best_match = best_bert_match
        jaccard_score = 0.0
        
        # If BERT fails, try Jaccard fallback with FULL OCR text (already spell-corrected)
        if not is_matched:
            # Concatenate ALL corrected OCR texts (reconstructed sentences)
            full_ocr_text = ' '.join(ocr_texts_corrected)
            
            # Calculate Jaccard similarity with full OCR text
            jaccard_similarity_full = calculate_jaccard_similarity(expected_text, full_ocr_text)
            
            # Check if Jaccard match is above fallback threshold
            if jaccard_similarity_full >= jaccard_fallback_threshold:
                is_matched = True
                match_method = 'Jaccard (fallback)'
                # For display, show the best BERT match but note it passed via Jaccard
                best_match = best_bert_match
                jaccard_score = jaccard_similarity_full
                word_coverage = 0.0
                words_found_count = 0
                words_expected_count = 0
            else:
                # Tier 3: Try word-level coverage (OCR already spell-corrected above)
                # Just check word intersection
                expected_words = set(expected_text.lower().split())
                # Use all corrected words from OCR
                ocr_word_set = set(word.lower() for word in all_corrected_words)
                words_found_count = len(expected_words.intersection(ocr_word_set))
                words_expected_count = len(expected_words)
                coverage = words_found_count / words_expected_count if words_expected_count > 0 else 0.0
                
                # Check if word coverage is above threshold
                if coverage >= word_coverage_threshold:
                    is_matched = True
                    match_method = 'Word-level (spell-corrected)'
                    best_match = best_bert_match
                    jaccard_score = jaccard_similarity_full
                    word_coverage = coverage
                else:
                    # All methods failed
                    match_method = None
                    jaccard_score = jaccard_similarity_full
                    word_coverage = coverage
        
        # Initialize word coverage variables for BERT-only matches
        if match_method == 'BERT':
            word_coverage = 0.0
            words_found_count = 0
            words_expected_count = 0
        
        if is_matched:
            matched_count += 1
        
        match_details.append({
            'expected_text': expected_text,
            'best_ocr_match': best_match,
            'bert_similarity': float(max_bert_similarity),
            'jaccard_similarity': float(jaccard_score) if jaccard_score > 0 else float(calculate_jaccard_similarity(expected_text, best_match)),
            'word_coverage': float(word_coverage),
            'words_found': int(words_found_count),
            'words_expected': int(words_expected_count),
            'match_method': match_method,
            'matched': is_matched
        })
    
    return matched_count, total_expected, match_details




def process_single_image(
    image_path: str, 
    image_id: int, 
    infographic_dir: str,
    ocr_model: PaddleOCR,
    bert_model: SentenceTransformer,
    spell_checker: SpellChecker,
    threshold: float = 0.9,
    jaccard_threshold: float = 0.6,
    word_coverage_threshold: float = 0.9
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
    
    # Extract text content from generated_infographic field using sophisticated extraction
    json_texts = []
    
    if 'generated_infographic' in infographic_entry:
        generated_text = infographic_entry['generated_infographic']
        json_texts = extract_text_elements(generated_text)
    
    # Perform OCR
    ocr_texts, ocr_word_count = perform_ocr(image_path, ocr_model)
    
    # Group OCR texts into sentences for better visibility
    ocr_sentences = group_ocr_texts_to_sentences(ocr_texts)
    
    # Match texts using BERT semantic similarity with Jaccard and word-level fallbacks
    matched_count, total_expected, match_details = match_texts_with_bert(
        json_texts, ocr_texts, bert_model, spell_checker, threshold, jaccard_threshold, word_coverage_threshold
    )
    
    print(f"  Text matching: {matched_count}/{total_expected} texts matched (BERT similarity >= {threshold:.2f})")
    
    # Determine if image should be filtered
    # Filter if not all expected texts are matched (require 100% match)
    should_filter = (matched_count < total_expected)
    
    if should_filter:
        # Return the infographic entry for failed images
        print(f"  ✗ Image {image_id} failed OCR check - will be added to failed.json for regeneration")
        
        # Create detailed reason
        unmatched = [d for d in match_details if not d['matched']]
        reason_parts = [f"Only {matched_count}/{total_expected} texts matched"]
        if unmatched:
            reason_parts.append(f"Unmatched texts: {', '.join([d['expected_text'][:30] + '...' if len(d['expected_text']) > 30 else d['expected_text'] for d in unmatched[:3]])}")
        reason = '; '.join(reason_parts)
        
        # Separate matched and unmatched texts
        matched_texts = []
        unmatched_texts = []
        
        for detail in match_details:
            if detail['matched']:
                matched_texts.append({
                    'expected_text': detail['expected_text'],
                    'ocr_match': detail['best_ocr_match'],
                    'bert_similarity': detail['bert_similarity'],
                    'jaccard_similarity': detail['jaccard_similarity'],
                    'word_coverage': detail['word_coverage'],
                    'words_found': detail['words_found'],
                    'words_expected': detail['words_expected'],
                    'match_method': detail['match_method']
                })
            else:
                unmatched_texts.append({
                    'expected_text': detail['expected_text'],
                    'best_ocr_attempt': detail['best_ocr_match'],
                    'bert_similarity': detail['bert_similarity'],
                    'jaccard_similarity': detail['jaccard_similarity'],
                    'word_coverage': detail['word_coverage'],
                    'words_found': detail['words_found'],
                    'words_expected': detail['words_expected']
                })
        
        return {
            'infographic_entry': infographic_entry,
            'image_id': image_id,
            'image_filename': os.path.basename(image_path),
            'matched_count': matched_count,
            'total_expected': total_expected,
            'ocr_sentences': ocr_sentences,
            'expected_texts': json_texts,
            'matched_texts': matched_texts,
            'unmatched_texts': unmatched_texts,
            'match_details': match_details,
            'reason': reason
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
        default=0.9,
        help='BERT similarity threshold (default: 0.9). Primary matching method using semantic similarity.'
    )
    parser.add_argument(
        '--jaccard-threshold',
        type=float,
        default=0.6,
        help='Jaccard similarity threshold for fallback (default: 0.6). Used when BERT fails, matches based on word overlap. Calculates % of expected words found in OCR.'
    )
    parser.add_argument(
        '--word-coverage-threshold',
        type=float,
        default=0.9,
        help='Word coverage threshold for spell-corrected matching (default: 0.9). Pass if >= 90%% words found after spell correction.'
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
    print(f"Using BERT similarity threshold: {args.threshold:.2f}")
    print(f"Using Jaccard fallback threshold: {args.jaccard_threshold:.2f}")
    print(f"Using word coverage threshold: {args.word_coverage_threshold:.2f}")
    print(f"Images directory: {args.images_dir}")
    print(f"Infographic directory: {args.infographic_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Initialize OCR model once (expensive operation)
    print("Initializing PaddleOCR model...")
    ocr_model = PaddleOCR(use_textline_orientation=True, lang='en')
    print("OCR model initialized successfully!")
    
    # Initialize BERT model
    print("Initializing sentence-transformers BERT model (all-MiniLM-L6-v2)...")
    bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("BERT model initialized successfully!")
    
    # Initialize SpellChecker
    print("Initializing spell checker...")
    spell_checker = SpellChecker()
    print("Spell checker initialized successfully!")
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
            bert_model,
            spell_checker,
            args.threshold,
            args.jaccard_threshold,
            args.word_coverage_threshold
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
            'bert_threshold': args.threshold,
            'jaccard_fallback_threshold': args.jaccard_threshold,
            'word_coverage_threshold': args.word_coverage_threshold,
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
                'matched_count': r.get('matched_count', 0),
                'total_expected': r.get('total_expected', 0),
                'reason': r['reason'],
                'ocr_sentences': r.get('ocr_sentences', []),
                'expected_texts': r.get('expected_texts', []),
                'matched_texts': r.get('matched_texts', []),
                'unmatched_texts': r.get('unmatched_texts', [])
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