import json
import os
import argparse
import glob
from typing import List, Dict, Any
import random
from pathlib import Path


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
                
                # Find all PNG files in this directory (exclude bbox files)
                png_files = glob.glob(os.path.join(narrator_dir, "*.png"))
                # Filter out bbox files - only keep original images
                png_files = [f for f in png_files if not f.endswith('_bbox.png')]
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


def convert_squad_v2_to_conversations(qa_data: Dict) -> List[Dict]:
    """
    Convert Squad v2 QA data to conversation format.
    
    Args:
        qa_data: Single entry from Squad v2 dataset
        
    Returns:
        List of conversation turns
    """
    conversations = []
    
    # Add context-based question if context exists
    context = qa_data.get('context', '').strip()
    question = qa_data.get('question', '').strip()
    
    if question:
        # Create a human question
        human_question = f"<image>\n{question}"
        conversations.append({
            "from": "human", 
            "value": human_question
        })
        
        # Get answer
        answers = qa_data.get('answers', {})
        if answers and 'text' in answers and len(answers['text']) > 0:
            # Use the first answer if multiple answers exist
            answer_text = answers['text'][0].strip()
        else:
            # Handle unanswerable questions in Squad v2
            # answer_text = "The question cannot be answered based on the given information."
            answer_text = ""
        
        conversations.append({
            "from": "gpt",
            "value": answer_text
        })
    
    # # Add additional context-based questions if context is available
    # if context and len(conversations) == 2:  # Only if we have the main Q&A
    #     # Generate additional questions based on context
    #     additional_questions = [
    #         "What is the main topic discussed in this infographic?",
    #         "Can you summarize the key information shown in the image?",
    #         "What are the main visual elements in this infographic?"
    #     ]
        
    #     # Randomly select 1-2 additional questions
    #     num_additional = random.randint(0, 2)
    #     selected_questions = random.sample(additional_questions, min(num_additional, len(additional_questions)))
        
    #     for add_question in selected_questions:
    #         conversations.append({
    #             "from": "human",
    #             "value": add_question
    #         })
    #         # Generate generic answers based on context
    #         if "main topic" in add_question.lower():
    #             conversations.append({
    #                 "from": "gpt", 
    #                 "value": f"This infographic discusses information related to the given context about {qa_data.get('title', 'the topic')}."
    #             })
    #         elif "summarize" in add_question.lower():
    #             conversations.append({
    #                 "from": "gpt",
    #                 "value": "This infographic presents key information in a visual format, combining text and graphical elements to convey the main points effectively."
    #             })
    #         elif "visual elements" in add_question.lower():
    #             conversations.append({
    #                 "from": "gpt",
    #                 "value": "The infographic contains various visual elements including text sections, graphical components, and structured layouts to present information clearly."
    #             })
    
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
        max_samples: Maximum number of samples to include
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    
    print(f"Loading QA data from {qa_file_path}")
    qa_data = load_jsonl(qa_file_path)
    
    print(f"Finding image files in {image_base_dir}/{dataset_name}")
    image_map = find_image_files(image_base_dir, dataset_name)
    
    if not image_map:
        print(f"No images found in {image_base_dir}/{dataset_name}")
        return
    
    training_data = []
    
    # Process each QA entry
    for idx, qa_entry in enumerate(qa_data):
        # For JSONL format: line number (0-based idx) + 1 = actual line number
        # But image filename uses the actual line number as image ID
        # So line 10070 (idx=10069) should map to image 10070.png
        image_idx = idx + 1
        
        if image_idx not in image_map:
            print(f"Warning: No image found for QA entry {idx+1} (image_idx: {image_idx})")
            continue
        
        image_path = image_map[image_idx]
        
        # Convert to conversations based on dataset type
        if dataset_type == "squad_v2":
            conversations = convert_squad_v2_to_conversations(qa_entry)
        else:
            conversations = convert_other_dataset_to_conversations(qa_entry, dataset_type)
        
        if not conversations:
            continue
        
        # Create training entry
        entry_id = qa_entry.get('id', f"{dataset_type}_{image_idx:06d}")
        training_entry = {
            "id": entry_id,
            "image": image_path,
            "conversations": conversations
        }
        
        # Log mapping for verification (first 3 entries)
        if len(training_data) < 3:
            print(f"Entry {len(training_data)+1}: Line {idx+1} -> Image {image_idx} -> {image_path}")
            print(f"  QA ID: {entry_id}")
            print(f"  Question: {qa_entry.get('question', '')[:100]}...")
        
        training_data.append(training_entry)
        
        # Check if we've reached max samples
        if max_samples and len(training_data) >= max_samples:
            break
    
    # Shuffle the data if seed is provided
    if seed is not None:
        random.shuffle(training_data)
    
    print(f"Created {len(training_data)} training samples")
    
    # Save the training dataset
    print(f"Saving training dataset to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_json(training_data, output_file)
    
    print(f"Training dataset saved successfully!")
    print(f"Total samples: {len(training_data)}")
    
    # Print some statistics
    if training_data:
        avg_conversations = sum(len(entry['conversations']) for entry in training_data) / len(training_data)
        print(f"Average conversations per sample: {avg_conversations:.2f}")
        
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
        max_samples=args.max_samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()