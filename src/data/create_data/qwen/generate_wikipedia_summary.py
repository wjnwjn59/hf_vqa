import json
import os
from pathlib import Path
from jinja2 import Template
from tqdm import tqdm
import argparse

# Import the Qwen3 inference module
from src.inference.qwen3_inference import Qwen3Inference

def load_summary_template(template_path):
    """Load the summary.jinja template"""
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    return Template(template_content)

def load_wikipedia_data(data_path):
    """Load the Wikipedia processed data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_chunk_to_file(chunk, output_dir, start_wiki_idx):
    """Save a chunk of results to file"""
    if not chunk:
        return None
        
    # Calculate file index based on first summary ID in chunk
    first_summary_id = chunk[0]['summary_id']
    file_index = (first_summary_id - 1) // 50 + 1  # Convert to 1-based file indexing
    
    filename = f"summarize{file_index:06d}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Saved {len(chunk)} summaries to {filename} (IDs: {chunk[0]['summary_id']}-{chunk[-1]['summary_id']})")
    return filename

def main():
    parser = argparse.ArgumentParser(description='Generate Wikipedia summaries using Qwen with vLLM')
    parser.add_argument('--model_name', type=str, default='unsloth/Qwen3-8B',
                        help='Model name or path')
    parser.add_argument('--input_data', type=str, 
                        default='/home/thinhnp/hf_vqa/src/data/create_data/wikipedia/wikipedia_full_processed.json',
                        help='Path to input Wikipedia full article data')
    parser.add_argument('--template_path', type=str,
                        default='/home/thinhnp/hf_vqa/src/prompts/summary.jinja',
                        help='Path to summary.jinja template')
    parser.add_argument('--output_path', type=str,
                        default='/home/thinhnp/hf_vqa/src/data/create_data/qwen/wikipedia_summaries.json',
                        help='Path to output JSON file')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None for all)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference (smaller due to longer texts)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=4096,
                        help='Maximum tokens to generate')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization')
    parser.add_argument('--start-wiki', type=int, default=0,
                        help='Start index for Wikipedia data processing (inclusive)')
    parser.add_argument('--end-wiki', type=int, default=None,
                        help='End index for Wikipedia data processing (exclusive)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for summary files (default: src/data/create_data/output/summarize)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Initializing Wikipedia Summary Generation")
    print("="*60)
    
    # Load template
    print(f"\n[1/5] Loading summary template from: {args.template_path}")
    template = load_summary_template(args.template_path)
    
    # Load Wikipedia data
    print(f"\n[2/5] Loading Wikipedia data from: {args.input_data}")
    wiki_data_full = load_wikipedia_data(args.input_data)
    print(f"Total entries loaded: {len(wiki_data_full)}")
    
    # Slice the Wikipedia data based on start and end indices
    end_idx = args.end_wiki if args.end_wiki is not None else len(wiki_data_full)
    wiki_data = wiki_data_full[args.start_wiki:end_idx]
    
    print(f"Processing slice [{args.start_wiki}:{end_idx}] = {len(wiki_data)} entries")
    
    # Limit samples if specified (applied after slicing)
    if args.max_samples:
        wiki_data = wiki_data[:args.max_samples]
        print(f"Further limited to first {args.max_samples} samples")
    
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
    print(f"\n[4/4] Generating Wikipedia summaries")
    print(f"Batch size: {args.batch_size}")
    print("-"*60)
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Default to output/summarize from repository root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, '../../../../'))
        output_dir = os.path.join(repo_root, 'src/data/create_data/output/summarize')
    
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
    total_batches = (len(wiki_data) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(0, len(wiki_data), args.batch_size), 
                          desc="Processing batches", total=total_batches):
        batch = wiki_data[batch_idx:batch_idx + args.batch_size]
        
        # Prepare prompts for batch
        batch_prompts = []
        for item in batch:
            # Use the full article text for summarization - limit to reasonable length for model
            source_text = item['text'][:8000]  # Limit to 8000 chars to avoid context overflow
            
            # Render the template with the source text
            rendered_prompt = template.render(source_text=source_text)
            batch_prompts.append(rendered_prompt)
        
        # Generate responses for batch (without thinking mode and without JSON parsing)
        try:
            responses = qwen_inference.generate(
                batch_prompts, 
                enable_thinking=False
            )
            
            # Process outputs
            for i, (item, response) in enumerate(zip(batch, responses)):
                # Add unique summary ID
                summary_id = args.start_wiki + total_processed + 1
                
                result = {
                    "id": item["id"],
                    "title": item["title"],
                    "categories": item.get("categories", []),
                    "generated_summary": response.strip(),
                    "success": True,
                    "summary_id": summary_id
                }
                
                results.append(result)
                total_processed += 1
                successful_count += 1
                
                # Save to file when we have enough results
                if len(results) >= chunk_size:
                    filename = save_chunk_to_file(results, output_dir, args.start_wiki)
                    if filename:
                        saved_files.append(filename)
                    results = []  # Clear the results list
                    
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            # Handle failed batch
            for i, item in enumerate(batch):
                summary_id = args.start_wiki + total_processed + 1
                
                result = {
                    "id": item["id"],
                    "title": item["title"],
                    "source_text": item["text"][:8000],
                    "categories": item.get("categories", []),
                    "generated_summary": "",
                    "success": False,
                    "error": str(e),
                    "summary_id": summary_id
                }
                
                results.append(result)
                total_processed += 1
                failed_count += 1
                
                # Save to file when we have enough results
                if len(results) >= chunk_size:
                    filename = save_chunk_to_file(results, output_dir, args.start_wiki)
                    if filename:
                        saved_files.append(filename)
                    results = []  # Clear the results list
    
    # Save any remaining results
    if results:
        print("\n" + "="*60)
        print("Saving final chunk")
        print("="*60)
        filename = save_chunk_to_file(results, output_dir, args.start_wiki)
        if filename:
            saved_files.append(filename)
    
    # Final statistics
    print("\n" + "="*60)
    print("Summary Generation Complete - Final Statistics")
    print("="*60)
    
    if total_processed > 0:
        first_id = args.start_wiki + 1
        last_id = args.start_wiki + total_processed
        print(f"Summary ID range: {first_id:06d} - {last_id:06d}")
    
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
    print("Summary generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()