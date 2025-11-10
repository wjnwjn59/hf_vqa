import json
import os
import argparse
import glob
from typing import List, Dict, Any
import random
from pathlib import Path


def load_wiki_files(wiki_dir: str) -> List[Dict[str, Any]]:
    """
    Load all wiki*.json files from the wiki directory.
    
    Args:
        wiki_dir: Directory containing wiki*.json files
        
    Returns:
        List of wiki entries with index and qa_pairs
    """
    wiki_files = sorted(glob.glob(os.path.join(wiki_dir, "wiki*.json")))
    
    if not wiki_files:
        print(f"Warning: No wiki files found in {wiki_dir}")
        return []
    
    all_entries = []
    total_qa_pairs = 0
    
    for wiki_file in wiki_files:
        print(f"Loading {os.path.basename(wiki_file)}...")
        with open(wiki_file, 'r', encoding='utf-8') as f:
            entries = json.load(f)
            
            for entry in entries:
                # Support both 'qa_pairs' and 'original_qa_pairs' field names
                qa_pairs = entry.get('qa_pairs') or entry.get('original_qa_pairs')
                if 'index' in entry and qa_pairs:
                    all_entries.append({
                        'index': entry['index'],
                        'qa_pairs': qa_pairs
                    })
                    total_qa_pairs += len(qa_pairs)
    
    print(f"Loaded {len(all_entries)} wiki entries from {len(wiki_files)} files")
    print(f"Total QA pairs: {total_qa_pairs}")
    return all_entries


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


def find_image_for_index(image_base_dir: str, dataset_name: str, index: int) -> str:
    """
    Find the image file for a given index.
    
    Args:
        image_base_dir: Base directory containing dataset outputs
        dataset_name: Name of the dataset (e.g., 'squad_v2')
        index: Image index from wiki file
    
    Returns:
        Relative image path in format "narrator000001/1.png" or None if not found
    """
    # Calculate which narrator folder the image should be in
    # Each folder contains 50 images: narrator000001 has images 1-50, narrator000002 has 51-100, etc.
    folder_num = ((index - 1) // 50) + 1
    folder_name = f"narrator{folder_num:06d}"
    
    # Construct the full path
    dataset_dir = os.path.join(image_base_dir, dataset_name)
    narrator_dir = os.path.join(dataset_dir, folder_name)
    image_file = os.path.join(narrator_dir, f"{index}.png")
    
    # Check if image exists
    if os.path.exists(image_file):
        # Return relative path without dataset prefix: "narrator000001/1.png"
        return f"{folder_name}/{index}.png"
    else:
        return None


def convert_qa_to_conversation(qa_data: Dict) -> List[Dict]:
    """
    Convert a single QA pair to conversation format.
    Each QA pair gets its own <image> token.
    
    Args:
        qa_data: Single QA pair from wiki file
        
    Returns:
        List of conversation turns (always 2 items: human + gpt)
    """
    question = qa_data.get('question', '').strip()
    
    if not question:
        return None
        
    # Get answer and check if it's answerable
    answers = qa_data.get('answers', {})
    if not answers or 'text' not in answers or len(answers['text']) == 0:
        # Skip unanswerable questions
        return None
        
    # Check if answer is empty or just whitespace
    answer_text = answers['text'][0].strip() if answers['text'] else ""
    if not answer_text:
        # Skip questions with empty answers
        return None
    
    # Each QA pair gets <image> token
    conversations = [
        {
            "from": "human",
            "value": f"<image>{question}"
        },
        {
            "from": "gpt",
            "value": answer_text
        }
    ]
    
    return conversations


def create_training_dataset(
    wiki_dir: str,
    image_base_dir: str, 
    dataset_name: str,
    output_file: str,
    output_jsonl_file: str = None,
    max_samples: int = None,
    seed: int = None
) -> None:
    """
    Create training dataset in Qwen2-VL format from wiki files.
    Each QA pair becomes a separate training sample with its own image reference.
    
    Args:
        wiki_dir: Directory containing wiki*.json files
        image_base_dir: Base directory containing generated images
        dataset_name: Name of dataset (subfolder name in image_base_dir)
        output_file: Output JSON file path
        output_jsonl_file: Output JSONL file path (optional, auto-generated if not provided)
        max_samples: Maximum number of samples to include
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    print(f"Loading wiki files from {wiki_dir}")
    wiki_entries = load_wiki_files(wiki_dir)
    
    if not wiki_entries:
        print(f"No wiki entries found in {wiki_dir}")
        return
    
    training_data = []
    skipped_no_image = 0
    skipped_no_qa = 0
    
    print(f"\nProcessing wiki entries...")
    print(f"Image base directory: {image_base_dir}/{dataset_name}")
    
    # Process each wiki entry
    for wiki_entry in wiki_entries:
        index = wiki_entry.get('index')
        qa_pairs = wiki_entry.get('qa_pairs', [])
        
        if not index:
            continue
        
        if not qa_pairs:
            skipped_no_qa += 1
            continue
        
        # Find the image for this index
        image_path = find_image_for_index(image_base_dir, dataset_name, index)
        
        if not image_path:
            skipped_no_image += 1
            if skipped_no_image <= 5:  # Only print first 5 warnings
                print(f"Warning: No image found for index {index}")
            continue
        
        # Create ONE training sample for EACH QA pair
        for qa_data in qa_pairs:
            conversations = convert_qa_to_conversation(qa_data)
            
            if not conversations:
                # Skip unanswerable or invalid QA pairs
                continue
            
            # Create training entry - one per QA pair
            qa_id = qa_data.get('id', f"wiki_{index}_{len(training_data)}")
            training_entry = {
                "id": qa_id,
                "image": image_path,
                "conversations": conversations
            }
            
            training_data.append(training_entry)
            
            # Log first few entries for verification
            if len(training_data) <= 3:
                print(f"\nSample {len(training_data)}:")
                print(f"  Index: {index}")
                print(f"  Image: {image_path}")
                print(f"  QA ID: {qa_id}")
                print(f"  Question: {conversations[0]['value'][:80]}...")
                print(f"  Answer: {conversations[1]['value'][:80]}...")
            
            # Check if we've reached max samples
            if max_samples and len(training_data) >= max_samples:
                break
        
        # Check if we've reached max samples
        if max_samples and len(training_data) >= max_samples:
            break
    
    print(f"\nProcessing complete!")
    print(f"  Total training samples created: {len(training_data)}")
    print(f"  Skipped (no image): {skipped_no_image}")
    print(f"  Skipped (no QA pairs): {skipped_no_qa}")
    
    # Shuffle the data if seed is provided
    if seed is not None:
        random.shuffle(training_data)
        print(f"  Data shuffled with seed: {seed}")
    
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
        description="Convert wiki files and infographic images to Qwen2-VL training format.\n"
                    "Each QA pair becomes a separate training sample."
    )
    
    parser.add_argument(
        "--wiki-dir",
        type=str,
        required=True,
        help="Directory containing wiki*.json files (e.g., src/data/narrator/wiki)"
    )
    
    parser.add_argument(
        "--image-base-dir", 
        type=str,
        default="./src/data/bizgen/output",
        help="Base directory containing generated images (default: ./src/data/bizgen/output)"
    )
    
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Dataset name (subfolder name in image-base-dir, e.g., 'squad_v2')"
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
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    create_training_dataset(
        wiki_dir=args.wiki_dir,
        image_base_dir=args.image_base_dir,
        dataset_name=args.dataset_name,
        output_file=args.output_file,
        output_jsonl_file=args.output_jsonl_file,
        max_samples=args.max_samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()