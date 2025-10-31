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

def group_qa_by_context(data):
    """
    Group QA pairs by context.
    
    Args:
        data: List of items with 'context', 'question', 'answer' fields
        
    Returns:
        List of items where each item has 'context' and 'qa_list' (list of Q&A pairs)
    """
    context_groups = {}
    
    for item in data:
        context = item['context']
        qa_pair = {
            'question': item['question'],
            'answer': item['answer']
        }
        
        if context not in context_groups:
            context_groups[context] = {
                'context': context,
                'qa_list': [],
                'ids': [],
                'title': item.get('title', None)
            }
        
        context_groups[context]['qa_list'].append(qa_pair)
        if item.get('id'):
            context_groups[context]['ids'].append(item['id'])
    
    # Convert to list and add summary info
    grouped_data = []
    for context, group_info in context_groups.items():
        group_info['qa_count'] = len(group_info['qa_list'])
        grouped_data.append(group_info)
    
    print(f"Grouped {len(data)} QA pairs into {len(grouped_data)} unique contexts")
    print(f"Average QA pairs per context: {len(data) / len(grouped_data):.2f}")
    
    return grouped_data

def load_input_data(dataset_type, input_path, deduplicate_context=False, group_by_context=False):
    """
    Load input data based on dataset type.
    For 'squad_v2', read from a single .jsonl file and extract 'context' and 'question'.
    For other types, fallback to summarize loader (to be defined).
    
    Args:
        dataset_type: Type of dataset ('squad_v2' or other)
        input_path: Path to the input file or directory
        deduplicate_context: If True, keep only unique contexts (for templates that don't need questions)
        group_by_context: If True, group QA pairs by context (for content_des_all.jinja template)
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
                    if deduplicate_context and not group_by_context:
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
        
        # Group by context if requested
        if group_by_context:
            all_data = group_qa_by_context(all_data)
            print(f"Final grouped data: {len(all_data)} unique contexts")
        elif deduplicate_context:
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
    parser.add_argument('--start', type=int, default=0,
                        help='Start index for data processing (inclusive)')
    parser.add_argument('--end', type=int, default=None,
                        help='End index for data processing (exclusive)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for infographic files (default: src/data/create_data/output/infographic)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Initializing Infographic Data Generation")
    print("="*60)
    
    # Load template
    print(f"\n[1/4] Loading bizgen template from: {args.template_path}")
    template = load_bizgen_template(args.template_path)
    
    # Determine template type and data processing strategy
    template_name = os.path.basename(args.template_path)
    deduplicate_context = template_name in ["bizgen_faithful_reproduction.jinja", "bizgen_design_drive_reproduction.jinja"]
    group_by_context = template_name == "content_des_all.jinja"
    requires_answer = template_name in ["bizgen_context_qa_full.jinja", "content_des_all.jinja"]
    
    if deduplicate_context:
        print(f"Template '{template_name}' detected - will deduplicate contexts")
    elif group_by_context:
        print(f"Template '{template_name}' detected - will group QA pairs by context")
    
    if requires_answer:
        print(f"Template '{template_name}' detected - will include answer field")
    
    # Load input data
    print(f"\n[2/4] Loading input data from: {args.input_data}")
    input_data_full = load_input_data(args.dataset_type, args.input_data, 
                                      deduplicate_context=deduplicate_context,
                                      group_by_context=group_by_context)
    print(f"Total entries loaded: {len(input_data_full)}")

    # Apply start and end slicing first
    end_idx = args.end if args.end is not None else len(input_data_full)
    input_data_sliced = input_data_full[args.start:end_idx]
    print(f"Sliced data from index {args.start} to {end_idx}: {len(input_data_sliced)} samples")

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
                template_name = os.path.basename(args.template_path)
                if template_name in ["bizgen_faithful_reproduction.jinja", "bizgen_design_drive_reproduction.jinja"]:
                    # Only use context
                    rendered_prompt = template.render(paragraph_input=item["context"])
                elif template_name == "bizgen_context_qa.jinja":
                    # Use context and question
                    rendered_prompt = template.render(paragraph_input=item["context"], qa_pairs=item["question"])
                elif template_name == "bizgen_context_qa_full.jinja":
                    # Use context, question, and answer
                    rendered_prompt = template.render(
                        context=item["context"], 
                        question=item["question"],
                        answer=item["answer"]
                    )
                elif template_name == "content_des_all.jinja":
                    # Use context and list of QA pairs
                    rendered_prompt = template.render(
                        context=item["context"],
                        qa_list=item["qa_list"]
                    )
                else:
                    # Default: just use context
                    rendered_prompt = template.render(paragraph_input=item["context"])
            else:
                # Fallback for other dataset types (define as needed)
                rendered_prompt = template.render(brief_input=item.get("generated_summary", ""))
            batch_prompts.append(rendered_prompt)

        if batch_idx == 0:
            print(batch_prompts[0])

        # Generate and parse responses for batch
        parsed_responses = qwen_inference.generate_and_parse_json(
            batch_prompts, 
            enable_thinking=False
        )

        # Process outputs
        for i, (item, parsed) in enumerate(zip(batch, parsed_responses)):
            # Calculate global infographic_id based on start index
            infographic_id = args.start + total_processed + 1
            
            # Base result structure
            result = {
                "generated_infographic": parsed["response"],
                "success": parsed["success"],
                "infographic_id": infographic_id
            }
            
            # Add different fields based on data structure
            if group_by_context:
                # For content_des_all.jinja: include context, qa_count, and ids
                result.update({
                    "context": item["context"],
                    "qa_count": item.get("qa_count", len(item.get("qa_list", []))),
                    "ids": item.get("ids", []),
                    "title": item.get("title", None)
                })
            else:
                # For other templates: include individual id and title
                result.update({
                    "id": item.get("id", None),
                    "title": item.get("title", None)
                })
            
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
        first_id = args.start + 1
        last_id = args.start + total_processed
        print(f"Infographic ID range: {first_id:06d} - {last_id:06d}")
    
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

