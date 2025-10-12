import json
import os
import glob
import argparse
from pathlib import Path
from jinja2 import Template
from tqdm import tqdm
from PIL import Image

# Import the QwenVL inference module
from src.inference.qwenvl_inference import QwenVLInference


def load_vqg_template(template_path):
    """Load the vqg.jinja template"""
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    return Template(template_content)


def get_image_files(images_dir):
    """Get all .png image files from the directory (excluding _bbox.png files)"""
    image_files = []
    
    # Get all .png files that don't end with _bbox.png
    pattern = os.path.join(images_dir, "*.png")
    all_png_files = glob.glob(pattern)
    
    for file_path in all_png_files:
        filename = os.path.basename(file_path)
        if not filename.endswith('_bbox.png'):
            # Extract ID from filename (e.g., "1.png" -> 1)
            try:
                image_id = int(filename.replace('.png', ''))
                image_files.append({
                    'id': image_id,
                    'path': file_path,
                    'filename': filename
                })
            except ValueError:
                print(f"Warning: Could not extract ID from filename: {filename}")
                continue
    
    # Sort by ID
    image_files.sort(key=lambda x: x['id'])
    
    return image_files


def main():
    parser = argparse.ArgumentParser(description='Generate VQA data using QwenVL with vLLM')
    parser.add_argument('--model_name', type=str, default='unsloth/Qwen2-VL-7B-Instruct',
                        help='Model name or path')
    parser.add_argument('--images_dir', type=str, 
                        default='/home/thinhnp/hf_vqa/src/data/bizgen/output/subset_0_516',
                        help='Path to directory containing images')
    parser.add_argument('--template_path', type=str,
                        default='/home/thinhnp/hf_vqa/src/prompts/vqg.jinja',
                        help='Path to vqg.jinja template')
    parser.add_argument('--output_path', type=str,
                        default=None,
                        help='Path to output JSON file (default: vqa_data.json in images_dir)')
    parser.add_argument('--num_questions', type=int, default=5,
                        help='Number of questions to generate per image')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=2048,
                        help='Maximum tokens to generate')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of images to process (None for all)')
    parser.add_argument('--start_id', type=int, default=None,
                        help='Start processing from this image ID (inclusive)')
    parser.add_argument('--end_id', type=int, default=None,
                        help='End processing at this image ID (inclusive)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Initializing VQA Data Generation")
    print("="*60)
    
    # Load template
    print(f"\n[1/4] Loading VQG template from: {args.template_path}")
    template = load_vqg_template(args.template_path)
    
    # Get image files
    print(f"\n[2/4] Loading images from: {args.images_dir}")
    image_files = get_image_files(args.images_dir)
    print(f"Total images found: {len(image_files)}")
    
    if not image_files:
        print("Error: No valid image files found!")
        return
    
    # Filter by ID range if specified
    if args.start_id is not None or args.end_id is not None:
        original_count = len(image_files)
        image_files = [
            img for img in image_files 
            if (args.start_id is None or img['id'] >= args.start_id) and
               (args.end_id is None or img['id'] <= args.end_id)
        ]
        print(f"Filtered to {len(image_files)} images (ID range: {args.start_id}-{args.end_id})")
    
    # Limit samples if specified
    if args.max_samples:
        image_files = image_files[:args.max_samples]
        print(f"Limited to first {args.max_samples} images")
    
    print(f"Processing {len(image_files)} images")
    print(f"ID range: {image_files[0]['id']} - {image_files[-1]['id']}")
    
    # Set output path - default to vqa_data.json in the images directory
    if args.output_path is None:
        args.output_path = os.path.join(args.images_dir, 'vqa_data.json')
    
    print(f"Output will be saved to: {args.output_path}")
    
    # Initialize QwenVL inference model
    print(f"\n[3/4] Initializing QwenVL inference model: {args.model_name}")
    qwenvl_inference = QwenVLInference(
        model_name=args.model_name,
        max_model_len=16384,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto",
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Generate VQA data
    print(f"\n[4/4] Generating VQA data")
    print(f"Questions per image: {args.num_questions}")
    print(f"Batch size: {args.batch_size}")
    print("-"*60)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    results = []
    total_processed = 0
    successful_count = 0
    failed_count = 0
    total_batches = (len(image_files) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(0, len(image_files), args.batch_size), 
                          desc="Processing batches", total=total_batches):
        batch_files = image_files[batch_idx:batch_idx + args.batch_size]
        
        # Prepare batch data
        batch_images = []
        batch_prompts = []
        
        for img_file in batch_files:
            # Load image
            try:
                image = Image.open(img_file['path']).convert('RGB')
                batch_images.append(image)
                
                # Render the template with num_questions
                rendered_prompt = template.render(num_questions=args.num_questions)
                batch_prompts.append(rendered_prompt)
                
            except Exception as e:
                print(f"Error loading image {img_file['path']}: {e}")
                # Add placeholder for failed image
                batch_images.append(None)
                batch_prompts.append("")
        
        # Filter out None images
        valid_indices = [i for i, img in enumerate(batch_images) if img is not None]
        if not valid_indices:
            # Skip this batch if no valid images
            for img_file in batch_files:
                result = {
                    "image_id": img_file['id'],
                    "image_filename": img_file['filename'],
                    "image_path": img_file['path'],
                    "vqa_pairs": [],
                    "success": False,
                    "error": "Failed to load image"
                }
                results.append(result)
                failed_count += 1
                total_processed += 1
            continue
        
        valid_images = [batch_images[i] for i in valid_indices]
        valid_prompts = [batch_prompts[i] for i in valid_indices]
        valid_files = [batch_files[i] for i in valid_indices]
        
        # Generate and parse responses for valid images
        try:
            parsed_responses = qwenvl_inference.generate_and_parse_json_batch(
                valid_images, valid_prompts
            )
            
            # Process results for valid images
            for i, (img_file, parsed) in enumerate(zip(valid_files, parsed_responses)):
                result = {
                    "image_id": img_file['id'],
                    "image_filename": img_file['filename'],
                    "image_path": img_file['path'],
                    "num_questions_requested": args.num_questions,
                    "success": parsed["success"]
                }
                
                if parsed["success"] and isinstance(parsed["response"], dict):
                    # Extract VQA pairs from the response
                    items = parsed["response"].get("items", [])
                    vqa_pairs = []
                    
                    for item in items:
                        if isinstance(item, dict) and "question" in item and "answer" in item:
                            vqa_pairs.append({
                                "question": item["question"],
                                "answer": item["answer"]
                            })
                    
                    result["vqa_pairs"] = vqa_pairs
                    result["num_questions_generated"] = len(vqa_pairs)
                    successful_count += 1
                else:
                    result["vqa_pairs"] = []
                    result["num_questions_generated"] = 0
                    result["error"] = parsed.get("error", "Unknown error")
                    result["raw_response"] = parsed["response"]
                    failed_count += 1
                
                results.append(result)
                total_processed += 1
            
            # Handle failed images in the original batch
            failed_indices = [i for i in range(len(batch_files)) if i not in valid_indices]
            for i in failed_indices:
                img_file = batch_files[i]
                result = {
                    "image_id": img_file['id'],
                    "image_filename": img_file['filename'],
                    "image_path": img_file['path'],
                    "vqa_pairs": [],
                    "success": False,
                    "error": "Failed to load image"
                }
                results.append(result)
                failed_count += 1
                total_processed += 1
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Mark all images in this batch as failed
            for img_file in batch_files:
                result = {
                    "image_id": img_file['id'],
                    "image_filename": img_file['filename'],
                    "image_path": img_file['path'],
                    "vqa_pairs": [],
                    "success": False,
                    "error": f"Batch processing error: {str(e)}"
                }
                results.append(result)
                failed_count += 1
                total_processed += 1
    
    # Sort results by image_id
    results.sort(key=lambda x: x['image_id'])
    
    # Save results to JSON file
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Calculate statistics
    total_questions_generated = sum(
        len(result.get("vqa_pairs", [])) for result in results if result["success"]
    )
    
    # Final statistics
    print("\n" + "="*60)
    print("Generation Complete - Final Statistics")
    print("="*60)
    print(f"Total images processed: {total_processed}")
    print(f"Successful: {successful_count} ({successful_count/total_processed*100:.1f}%)")
    print(f"Failed: {failed_count} ({failed_count/total_processed*100:.1f}%)")
    print(f"Total VQA pairs generated: {total_questions_generated}")
    if successful_count > 0:
        print(f"Average questions per successful image: {total_questions_generated/successful_count:.1f}")
    print(f"Output file: {args.output_path}")
    
    # Save summary statistics
    summary = {
        "total_images_processed": total_processed,
        "successful_images": successful_count,
        "failed_images": failed_count,
        "success_rate": successful_count/total_processed*100 if total_processed > 0 else 0,
        "total_vqa_pairs": total_questions_generated,
        "average_questions_per_image": total_questions_generated/successful_count if successful_count > 0 else 0,
        "parameters": {
            "model_name": args.model_name,
            "num_questions_per_image": args.num_questions,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens
        }
    }
    
    summary_path = args.output_path.replace('.json', '_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    print("\n" + "="*60)
    print("VQA Generation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()