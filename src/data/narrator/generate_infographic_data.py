import json
import os
from pathlib import Path
from jinja2 import Template
from tqdm import tqdm
import argparse

# Import the Qwen3 inference module
from src.inference.qwen3_inference import Qwen3Inference

def load_bizgen_template(template_path):
    """Load the bizgen.jinja template"""
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    return Template(template_content)

def load_input_data(dataset_type, input_path, deduplicate_context=False):
    """
    Load input data based on dataset type.
    For 'squad_v2', read from a single .jsonl file and extract 'context' and 'question'.
    For other types, fallback to summarize loader (to be defined).
    
    Args:
        dataset_type: Type of dataset ('squad_v2' or other)
        input_path: Path to the input file or directory
        deduplicate_context: If True, keep only unique contexts (for templates that don't need questions)
    """
    if dataset_type == "squad_v2":
        all_data = []
        seen_contexts = set()
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Only keep entries with both context and question
                if 'context' in item and 'question' in item:
                    context = item['context']
                    
                    # If deduplication is enabled, skip duplicate contexts
                    if deduplicate_context:
                        if context in seen_contexts:
                            continue
                        seen_contexts.add(context)
                    
                    all_data.append({
                        'context': context,
                        'question': item['question'],
                        'answer': item.get('answer', ''),
                        'id': item.get('id', None),
                        'title': item.get('title', None)
                    })
        
        if deduplicate_context:
            print(f"Loaded {len(all_data)} unique contexts from Squad v2 file: {input_path}")
        else:
            print(f"Loaded {len(all_data)} entries from Squad v2 file: {input_path}")
        return all_data
    else:
        # Fallback: original summarize loader (define as needed)
        summarize_files = sorted([f for f in os.listdir(input_path) if f.startswith('summarize') and f.endswith('.json')])
        all_data = []
        for filename in summarize_files:
            filepath = os.path.join(input_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                all_data.extend(chunk_data)
        print(f"Loaded {len(all_data)} summarize entries from {len(summarize_files)} files")
        return all_data

def save_chunk_to_file(chunk, output_dir, file_index):
    """Save a chunk of results to file"""
    if not chunk:
        return None
    
    filename = f"infographic{file_index:06d}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Saved {len(chunk)} infographics to {filename} (IDs: {chunk[0]['infographic_id']}-{chunk[-1]['infographic_id']})")
    return filename

def main():
    parser = argparse.ArgumentParser(description='Generate infographic data using Qwen with vLLM')
    parser.add_argument('--model_name', type=str, default='unsloth/Qwen3-8B',
                        help='Model name or path')
    parser.add_argument('--input_data', type=str, 
                        default='/mnt/VLAI_data/Squad_v2/squad_v2_train.jsonl',
                        help='Path to input data file or directory')
    parser.add_argument('--dataset_type', type=str, default='squad_v2',
                        help='Type of dataset: squad_v2 or summarize')
    parser.add_argument('--template_path', type=str,
                        default='/home/thinhnp/hf_vqa/src/prompts/bizgen.jinja',
                        help='Path to bizgen.jinja template')
    parser.add_argument('--output_path', type=str,
                        default='/home/thinhnp/hf_vqa/src/data/create_data/qwen/infographic_generated.json',
                        help='Path to output JSON file')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None for all)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=8192,
                        help='Maximum tokens to generate')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to process (None for all)')
    parser.add_argument('--start', type=int, default=1,
                        help='Start file index for data processing (inclusive, 1-based)')
    parser.add_argument('--end', type=int, default=None,
                        help='End file index for data processing (exclusive, 1-based)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for infographic files (default: src/data/create_data/output/infographic)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Initializing Infographic Data Generation")
    print("="*60)
    
    # Load template
    print(f"\n[1/5] Loading bizgen template from: {args.template_path}")
    template = load_bizgen_template(args.template_path)
    
    # Determine if we need to deduplicate contexts based on template type
    template_name = os.path.basename(args.template_path)
    deduplicate_context = template_name in ["bizgen_faithful_reproduction.jinja", "bizgen_design_drive_reproduction.jinja"]
    
    if deduplicate_context:
        print(f"Template '{template_name}' detected - will deduplicate contexts")
    
    # Check if template requires answer field
    requires_answer = template_name == "bizgen_context_qa_full.jinja"
    if requires_answer:
        print(f"Template '{template_name}' detected - will include answer field")
    
    # Load input data
    print(f"\n[2/5] Loading input data from: {args.input_data}")
    input_data_full = load_input_data(args.dataset_type, args.input_data, deduplicate_context=deduplicate_context)
    print(f"Total entries loaded: {len(input_data_full)}")

    # Convert file indices to data indices (each file contains 50 images)
    # File index is 1-based, data index is 0-based
    start_data_idx = (args.start - 1) * 50
    end_data_idx = (args.end - 1) * 50 if args.end is not None else len(input_data_full)
    
    # Validate file indices
    max_files_needed = (len(input_data_full) + 49) // 50  # Round up
    if args.start < 1:
        raise ValueError("Start file index must be >= 1")
    if args.end is not None and args.end <= args.start:
        raise ValueError("End file index must be > start file index")
    if args.start > max_files_needed:
        raise ValueError(f"Start file index {args.start} exceeds available data (max files: {max_files_needed})")
    
    print(f"File indices: {args.start} to {args.end if args.end else 'end'}")
    print(f"Data indices: {start_data_idx} to {end_data_idx}")
    print(f"Images per file: 50")

    # Apply start and end slicing first
    input_data_sliced = input_data_full[start_data_idx:end_data_idx]
    print(f"Sliced data from data index {start_data_idx} to {end_data_idx}: {len(input_data_sliced)} samples")

    # Then apply num_samples limit if specified
    if args.num_samples:
        input_data = input_data_sliced[:args.num_samples]
        print(f"Further limited to {args.num_samples} samples")
    else:
        input_data = input_data_sliced
        print(f"Processing all {len(input_data)} samples from slice")
    
    # Initialize Qwen3 inference model
    print(f"\n[3/4] Initializing Qwen3 inference model: {args.model_name}")
    qwen_inference = Qwen3Inference(
        model_name=args.model_name,
        max_model_len=32768,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="auto",
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    # Generate prompts
    print(f"\n[4/4] Generating infographic descriptions")
    print(f"Batch size: {args.batch_size}")
    print("-"*60)
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to output/infographic from repository root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, '../../../'))
        output_dir = os.path.join(repo_root, 'src/data/create_data/output/infographic')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize variables for incremental saving
    results = []
    saved_files = []
    chunk_size = 50
    total_processed = 0
    successful_count = 0
    failed_count = 0
    total_batches = (len(input_data) + args.batch_size - 1) // args.batch_size

    for batch_idx in tqdm(range(0, len(input_data), args.batch_size), 
                          desc="Processing batches", total=total_batches):
        batch = input_data[batch_idx:batch_idx + args.batch_size]

        # Prepare prompts for batch
        batch_prompts = []
        for item in batch:
            # Switch-case for prompt construction
            if args.dataset_type == "squad_v2":
                # For bizgen_faithful_reproduction.jinja or bizgen_design_drive_reproduction.jinja, only use context
                # For bizgen_context_qa.jinja, use context and question
                # For bizgen_context_qa_full.jinja, use context, question, and answer
                # Detect template type by filename
                template_name = os.path.basename(args.template_path)
                if template_name in ["bizgen_faithful_reproduction.jinja", "bizgen_design_drive_reproduction.jinja"]:
                    rendered_prompt = template.render(paragraph_input=item["context"])
                elif template_name == "bizgen_context_qa.jinja":
                    rendered_prompt = template.render(paragraph_input=item["context"], qa_pairs=item["question"])
                elif template_name == "bizgen_context_qa_full.jinja":
                    rendered_prompt = template.render(
                        context=item["context"], 
                        question=item["question"],
                        answer=item["answer"]
                    )
                else:
                    # Default: just use context
                    rendered_prompt = template.render(paragraph_input=item["context"])
            else:
                # Fallback for other dataset types (define as needed)
                rendered_prompt = template.render(brief_input=item.get("generated_summary", ""))
            batch_prompts.append(rendered_prompt)

        # Generate and parse responses for batch
        parsed_responses = qwen_inference.generate_and_parse_json(
            batch_prompts, 
            enable_thinking=False
        )

        # Process outputs
        for i, (item, parsed) in enumerate(zip(batch, parsed_responses)):
            # Calculate global infographic_id based on start data index
            infographic_id = start_data_idx + total_processed + 1
            result = {
                "id": item.get("id", None),
                "title": item.get("title", None),
                "generated_infographic": parsed["response"],
                "success": parsed["success"],
                "infographic_id": infographic_id
            }
            if not parsed["success"]:
                result["error"] = parsed["error"]
                failed_count += 1
            else:
                successful_count += 1
            results.append(result)
            total_processed += 1
            # Save to file when we have enough results
            if len(results) >= chunk_size:
                # Calculate file index based on first infographic_id in chunk
                first_infographic_id = results[0]['infographic_id']
                file_index = (first_infographic_id - 1) // chunk_size + 1
                filename = save_chunk_to_file(results, output_dir, file_index)
                if filename:
                    saved_files.append(filename)
                results = []  # Clear the results list
    
    # Save any remaining results
    if results:
        print("\n" + "="*60)
        print("Saving final chunk")
        print("="*60)
        # Calculate file index based on first infographic_id in chunk
        first_infographic_id = results[0]['infographic_id']
        file_index = (first_infographic_id - 1) // chunk_size + 1
        filename = save_chunk_to_file(results, output_dir, file_index)
        if filename:
            saved_files.append(filename)
    
    # Final statistics
    print("\n" + "="*60)
    print("Generation Complete - Final Statistics")
    print("="*60)
    
    if total_processed > 0:
        first_id = start_data_idx + 1
        last_id = start_data_idx + total_processed
        print(f"Infographic ID range: {first_id:06d} - {last_id:06d}")
        print(f"File index range: {args.start} - {args.end if args.end else args.start + (total_processed + 49) // 50}")
    
    print(f"Total samples processed: {total_processed}")
    print(f"Total files saved: {len(saved_files)}")
    print(f"Successful: {successful_count} ({successful_count/total_processed*100:.1f}%)")
    print(f"Failed: {failed_count} ({failed_count/total_processed*100:.1f}%)")
    print(f"Output directory: {output_dir}")
    
    # Save a summary of failed cases if any
    if failed_count > 0:
        print(f"\nNote: Failed cases were distributed across multiple files.")
        print(f"Check individual files for 'success': false entries.")
    
    print(f"\nFiles saved:")
    for filename in saved_files:
        print(f"  - {filename}")
    
    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()

