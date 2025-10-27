import json
import os
import argparse
import glob
from typing import List, Dict, Any
import random
from pathlib import Path


def load_squad_v2_data_deduplicated(input_path: str) -> List[Dict[str, Any]]:
    """Load Squad v2 data using the same deduplication logic as generate_narrator_with_bbox.py"""
    all_data = []
    seen_contexts = {}  # Map context to list of QA pairs
    total_entries = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            total_entries += 1
            
            context = entry.get('context', '').strip()
            if not context:
                continue
                
            # Create QA pair
            qa_pair = {
                'question': entry.get('question', ''),
                'answers': entry.get('answers', {}),
                'id': entry.get('id', ''),
                'is_impossible': entry.get('is_impossible', False)
            }
            
            # Add to context mapping
            if context not in seen_contexts:
                seen_contexts[context] = {
                    'context': context,
                    'qa_pairs': []
                }
            seen_contexts[context]['qa_pairs'].append(qa_pair)
    
    # Convert to list format
    for context_data in seen_contexts.values():
        all_data.append(context_data)
    
    total_qa_pairs = sum(len(item['qa_pairs']) for item in all_data)
    print(f"Loaded {total_entries} total entries from Squad v2 file: {input_path}")
    print(f"Unique contexts: {len(all_data)} (deduplication removed {total_entries - len(all_data)} entries)")
    print(f"Total QA pairs: {total_qa_pairs}")
    return all_data


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def load_json(file_path: str) -> Any:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str):
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file (one JSON object per line)."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def find_image_files(image_base_dir: str, dataset_name: str) -> Dict[str, str]:
    """
    Find all image files in the dataset directory.
    
    Args:
        image_base_dir: Base directory containing dataset outputs
        dataset_name: Name of the dataset (e.g., 'squad_v2')
    
    Returns:
        Dictionary mapping image indices to image file paths
    """
    image_map = {}
    dataset_dir = os.path.join(image_base_dir, dataset_name)
    
    if not os.path.exists(dataset_dir):
        print(f"Warning: Dataset directory {dataset_dir} does not exist")
        return image_map
    
    # Find all narrator* subdirectories
    narrator_dirs = glob.glob(os.path.join(dataset_dir, "narrator*"))
    narrator_dirs.sort()
    
    for narrator_dir in narrator_dirs:
        # Extract directory number to determine image index range
        dir_name = os.path.basename(narrator_dir)
        if dir_name.startswith('narrator'):
            try:
                dir_num = int(dir_name.replace('narrator', '').lstrip('0') or '0')
                
                # Each directory contains 50 images, starting from (dir_num-1)*50 + 1
                start_idx = (dir_num - 1) * 50 + 1
                
                # Find all PNG files in this directory (exclude bbox and faults files)
                png_files = glob.glob(os.path.join(narrator_dir, "*.png"))
                # Filter out bbox files and faults files - only keep original images
                png_files = [f for f in png_files if not f.endswith('_bbox.png') and not f.endswith('_faults.png')]
                png_files.sort()
                
                for i, png_file in enumerate(png_files):
                    # Extract the image number from filename (e.g., 10070 from 10070.png)
                    filename = os.path.basename(png_file)
                    if filename.endswith('.png'):
                        try:
                            image_number = int(filename[:-4])  # Remove .png extension
                            # Store relative path from dataset directory
                            rel_path = os.path.relpath(png_file, image_base_dir)
                            image_map[image_number] = rel_path
                        except ValueError:
                            # Skip files that don't have numeric names
                            continue
                    
            except ValueError:
                continue
    
    print(f"Found {len(image_map)} images in {dataset_dir}")
    return image_map


def convert_squad_v2_to_conversations(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Convert Squad v2 QA pairs to conversation format.
    Supports multiple QA pairs for the same context.
    
    Args:
        qa_pairs: List of QA pairs from Squad v2 dataset for the same context
        
    Returns:
        List of conversation turns
    """
    conversations = []
    
    # Process each QA pair for this context
    for qa_data in qa_pairs:
        question = qa_data.get('question', '').strip()
        
        if not question:
            continue
            
        # Get answer and check if it's answerable
        answers = qa_data.get('answers', {})
        if not answers or 'text' not in answers or len(answers['text']) == 0:
            # Skip unanswerable questions
            continue
            
        # Check if answer is empty or just whitespace
        answer_text = answers['text'][0].strip() if answers['text'] else ""
        if not answer_text:
            # Skip questions with empty answers
            continue
        
        # Create a human question
        # Only add <image> token for the first question
        if len(conversations) == 0:
            human_question = f"<image>\n{question}"
        else:
            human_question = question
            
        conversations.append({
            "from": "human", 
            "value": human_question
        })
        
        conversations.append({
            "from": "gpt",
            "value": answer_text
        })
    
    return conversations


def convert_other_dataset_to_conversations(qa_data: Dict, dataset_type: str) -> List[Dict]:
    """
    Convert other dataset formats to conversation format.
    Add more dataset types here as needed.
    
    Args:
        qa_data: Single entry from dataset
        dataset_type: Type of dataset
        
    Returns:
        List of conversation turns
    """
    conversations = []
    
    if dataset_type == "narrativeqa":
        # Handle NarrativeQA format
        question = qa_data.get('question', '').strip()
        answer = qa_data.get('answer', '').strip()
        
        if question and answer:
            conversations.append({
                "from": "human",
                "value": f"<image>\n{question}"
            })
            conversations.append({
                "from": "gpt", 
                "value": answer
            })
    
    elif dataset_type == "adversarialqa":
        # Handle AdversarialQA format
        question = qa_data.get('question', '').strip()
        answers = qa_data.get('answers', {})
        
        if question:
            conversations.append({
                "from": "human",
                "value": f"<image>\n{question}"
            })
            
            if answers and 'text' in answers and len(answers['text']) > 0:
                answer_text = answers['text'][0].strip()
            else:
                answer_text = "The question cannot be answered based on the given information."
            
            conversations.append({
                "from": "gpt",
                "value": answer_text
            })
    
    # Add more dataset types here as needed
    # elif dataset_type == "custom_dataset":
    #     # Handle custom dataset format
    #     pass
    
    return conversations


def create_training_dataset(
    qa_file_path: str,
    image_base_dir: str, 
    dataset_name: str,
    dataset_type: str,
    output_file: str,
    output_jsonl_file: str = None,
    max_samples: int = None,
    seed: int = None
) -> None:
    """
    Create training dataset in Qwen2-VL format.
    
    Args:
        qa_file_path: Path to QA data file (JSONL format)
        image_base_dir: Base directory containing generated images
        dataset_name: Name of dataset (subfolder name)
        dataset_type: Type of dataset for proper parsing
        output_file: Output JSON file path
        output_jsonl_file: Output JSONL file path (optional, auto-generated if not provided)
        max_samples: Maximum number of samples to include
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    print(f"Loading QA data from {qa_file_path}")
    if dataset_type == "squad_v2":
        # Use deduplicated loading for Squad v2
        qa_data = load_squad_v2_data_deduplicated(qa_file_path)
        is_deduplicated_format = True
    else:
        # Use regular loading for other formats
        qa_data = load_jsonl(qa_file_path)
        is_deduplicated_format = False
    
    print(f"Finding image files in {image_base_dir}/{dataset_name}")
    image_map = find_image_files(image_base_dir, dataset_name)
    
    if not image_map:
        print(f"No images found in {image_base_dir}/{dataset_name}")
        return
    
    training_data = []
    
    # Check format type
    if is_deduplicated_format or (qa_data and 'qa_pairs' in qa_data[0]):
        # Deduplicated format: each entry has context and multiple qa_pairs
        print("Using deduplicated Squad v2 format (context + qa_pairs)")
        
        for idx, context_entry in enumerate(qa_data):
            image_idx = idx + 1  # Use position in deduplicated data as image index
            
            if image_idx not in image_map:
                print(f"Warning: No image found for context {idx+1} (image_idx: {image_idx})")
                continue
            
            image_path = image_map[image_idx]
            qa_pairs = context_entry.get('qa_pairs', [])
            
            if not qa_pairs:
                continue
            
            # Convert to conversations based on dataset type
            if dataset_type == "squad_v2":
                conversations = convert_squad_v2_to_conversations(qa_pairs)
            else:
                # For other dataset types, process first QA pair only
                conversations = convert_other_dataset_to_conversations(qa_pairs[0], dataset_type)
            
            # Skip if no valid conversations (all unanswerable)
            if not conversations:
                print(f"Skipping context {idx+1}: no answerable questions")
                continue
            
            # Create training entry using first QA pair's ID or generate one
            first_qa = qa_pairs[0]
            entry_id = first_qa.get('id', f"{dataset_type}_{image_idx:06d}")
            training_entry = {
                "id": entry_id,
                "image": image_path,
                "conversations": conversations
            }
            
            # Log mapping for verification (first 3 entries)
            if len(training_data) < 3:
                print(f"Entry {len(training_data)+1}: Context {idx+1} -> Image {image_idx} -> {image_path}")
                print(f"  Total QA pairs: {len(qa_pairs)}")
                print(f"  Answerable QA pairs: {len(conversations)//2}")
                print(f"  Context preview: {context_entry.get('context', '')[:100]}...")
            
            training_data.append(training_entry)
            
            # Check if we've reached max samples
            if max_samples and len(training_data) >= max_samples:
                break
                
    else:
        # Original format: each entry is a single QA pair
        # Need to group by context and create one training entry per unique context
        print("Detected original Squad v2 format (individual QA entries)")
        print("Grouping QA pairs by context...")
        
        # Group QA pairs by context
        context_groups = {}
        for idx, qa_entry in enumerate(qa_data):
            context = qa_entry.get('context', '').strip()
            if context not in context_groups:
                context_groups[context] = {
                    'qa_pairs': [],
                    'first_idx': idx  # Track the first occurrence for image mapping
                }
            context_groups[context]['qa_pairs'].append(qa_entry)
        
        print(f"Found {len(context_groups)} unique contexts")
        
        # Process each unique context
        for context_idx, (context, context_data) in enumerate(context_groups.items()):
            # Calculate image_idx using the same logic as generate_narrator_with_bbox.py
            # infographic_id = start_data_idx + i + 1, where start_data_idx = 0 for full dataset
            # and i is the context index in the unique contexts list
            image_idx = context_idx + 1  # Use context position in unique contexts, not original data position
            
            if image_idx not in image_map:
                print(f"Warning: No image found for context {context_idx+1} (image_idx: {image_idx})")
                continue
            
            image_path = image_map[image_idx]
            qa_pairs = context_data['qa_pairs']
            
            # Convert to conversations based on dataset type
            if dataset_type == "squad_v2":
                conversations = convert_squad_v2_to_conversations(qa_pairs)
            else:
                # For other dataset types, process first QA pair only
                conversations = convert_other_dataset_to_conversations(qa_pairs[0], dataset_type)
            
            # Skip if no valid conversations (all unanswerable)
            if not conversations:
                print(f"Skipping context {context_idx+1}: no answerable questions")
                continue
            
            # Create training entry using first QA pair's ID or generate one
            first_qa = qa_pairs[0]
            entry_id = first_qa.get('id', f"{dataset_type}_{image_idx:06d}")
            training_entry = {
                "id": entry_id,
                "image": image_path,
                "conversations": conversations
            }
            
            # Log mapping for verification (first 3 entries)
            if len(training_data) < 3:
                print(f"Entry {len(training_data)+1}: Context {context_idx+1} -> Image {image_idx} -> {image_path}")
                print(f"  Total QA pairs: {len(qa_pairs)}")
                print(f"  Answerable QA pairs: {len(conversations)//2}")
                print(f"  First question: {first_qa.get('question', '')[:100]}...")
            
            training_data.append(training_entry)
            
            # Check if we've reached max samples
            if max_samples and len(training_data) >= max_samples:
                break
    
    # Shuffle the data if seed is provided
    if seed is not None:
        random.shuffle(training_data)
    
    print(f"Created {len(training_data)} training samples")
    
    # Generate JSONL file path if not provided
    if output_jsonl_file is None:
        # Convert JSON path to JSONL path (replace .json with .jsonl)
        if output_file.endswith('.json'):
            output_jsonl_file = output_file[:-5] + '.jsonl'
        else:
            output_jsonl_file = output_file + '.jsonl'
    
    # Save the training dataset in both formats
    print(f"Saving training dataset to {output_file} (JSON format)")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_json(training_data, output_file)
    
    print(f"Saving training dataset to {output_jsonl_file} (JSONL format)")
    os.makedirs(os.path.dirname(output_jsonl_file), exist_ok=True)
    save_jsonl(training_data, output_jsonl_file)
    
    print(f"Training datasets saved successfully!")
    print(f"JSON file: {output_file}")
    print(f"JSONL file: {output_jsonl_file}")
    print(f"Total samples: {len(training_data)}")
    
    # Print some statistics
    if training_data:
        total_conversations = sum(len(entry['conversations']) for entry in training_data)
        avg_conversations = total_conversations / len(training_data)
        print(f"Total conversation turns: {total_conversations}")
        print(f"Average conversations per sample: {avg_conversations:.2f}")
        
        # Count QA pairs (each pair = human + gpt turn)
        total_qa_pairs = total_conversations // 2
        print(f"Total QA pairs: {total_qa_pairs}")
        
        # Sample entry for verification
        print("\nSample entry:")
        print(json.dumps(training_data[0], indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description="Convert infographic images and QA data to Qwen2-VL training format"
    )
    
    parser.add_argument(
        "--qa-file",
        type=str,
        required=True,
        help="Path to QA data file (JSONL format)"
    )
    
    parser.add_argument(
        "--image-base-dir", 
        type=str,
        default="/home/thinhnp/hf_vqa/src/data/bizgen/output",
        help="Base directory containing generated images (default: /home/thinhnp/hf_vqa/src/data/bizgen/output)"
    )
    
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Dataset name (subfolder name in image-base-dir, e.g., 'squad_v2')"
    )
    
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        choices=["squad_v2", "narrativeqa", "adversarialqa", "chemlit", "mesaqa"],
        help="Type of dataset for proper parsing"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    
    parser.add_argument(
        "--output-jsonl-file",
        type=str,
        default=None,
        help="Output JSONL file path (optional, auto-generated from JSON path if not provided)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to include"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    create_training_dataset(
        qa_file_path=args.qa_file,
        image_base_dir=args.image_base_dir,
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        output_file=args.output_file,
        output_jsonl_file=args.output_jsonl_file,
        max_samples=args.max_samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()