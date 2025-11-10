import argparse
import json
import os
import csv
from pathlib import Path
from typing import Dict, List, Any, Callable
import importlib.util
from tqdm import tqdm
from collections import Counter
import torch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_CONFIGS
from typing import Optional

try:
    from torch.profiler import profile, ProfilerActivity
    _PROFILER_AVAILABLE = True
except Exception:
    _PROFILER_AVAILABLE = False



def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def most_common_answer(answers: List[Dict[str, str]]) -> str:
    """Extract the most common answer from a list of answer dictionaries."""
    return Counter(answer['answer'] for answer in answers).most_common(1)[0][0]


def load_narrator_qas_cache(dataset_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    Load all QAS files from NARRATOR dataset into memory cache.
    
    Args:
        dataset_dir: Base directory containing qas/ subdirectory
        
    Returns:
        Dict mapping image_id to QAS data containing:
            - context: str
            - original_qa_pairs: List[Dict]
            - generated_qa_pairs: List[Dict]
    """
    qas_cache = {}
    qas_dir = Path(dataset_dir) / "qas"
    
    if not qas_dir.exists():
        raise FileNotFoundError(f"QAS directory not found: {qas_dir}")
    
    for qas_file in qas_dir.glob("*.json"):
        image_id = int(qas_file.stem)
        with open(qas_file, 'r', encoding='utf-8') as f:
            qas_cache[image_id] = json.load(f)
    
    return qas_cache


def load_inference_module(model_name: str) -> Callable:
    """Dynamically load inference function from model file."""
    model_file = Path(__file__).parent / "models" / f"{model_name}.py"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    spec = importlib.util.spec_from_file_location(model_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {model_name}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inference


def extract_image_id(item: Dict[str, Any], dataset: str) -> Any:
    """Extract image ID based on dataset format."""
    # NARRATOR dataset has image_id directly in metadata
    if dataset == "narrator_val":
        return item["image_id"]
    
    image_name = item["image"]
    
    if dataset == "vqav2_val":
        return int(image_name.split("_")[-1].split(".")[0])
    elif dataset == "infographicvqa_val":
        return int(image_name.split(".")[0])
    else:  # textvqa_val, chartqa, docvqa_val
        return image_name.split("/")[-1].split(".")[0]


def process_batch_data(data: List[Dict], dataset: str, config, qas_cache: Optional[Dict[int, Dict[str, Any]]] = None) -> tuple:
    """Process and extract batch data efficiently."""
    qids = [item["question_id"] if "question_id" in item else item["id"] for item in data]
    img_ids = [extract_image_id(item, dataset) for item in data]
    
    # Handle NARRATOR dataset format
    if dataset == "narrator_val":
        if qas_cache is None:
            raise ValueError("QAS cache is required for NARRATOR dataset")
        
        questions = []
        gts = []
        img_paths = []
        
        for item in data:
            image_id = item["image_id"]
            qa_type = item["qa_type"]
            qa_index = item["qa_index"]
            
            # Fetch QA data from cache
            qas_data = qas_cache[image_id]
            qa_pairs_key = f"{qa_type}_qa_pairs"
            qa = qas_data[qa_pairs_key][qa_index]
            
            # Extract question and first answer text
            questions.append(qa["question"])
            
            # Handle cases where answers.text might be empty (unanswerable questions)
            if qa["answers"]["text"]:
                gts.append(qa["answers"]["text"][0])
            else:
                # For unanswerable questions, use empty string or "unanswerable"
                gts.append("")
            
            # Build image path: {image_id}.png
            img_paths.append(os.path.join(config.img_folder, f"{image_id}.png"))
        
        return qids, img_ids, questions, gts, img_paths
    
    # Handle other dataset formats
    is_chartqapro = dataset == "chartqapro_test"
    questions = [item["question"][0] if is_chartqapro else item["question"] for item in data]
    gts = [item["answer"][0] if is_chartqapro else item["answer"] for item in data]
    
    img_paths = [os.path.join(config.img_folder, item["image"]) for item in data]
    
    return qids, img_ids, questions, gts, img_paths


def save_results(data: List[Dict], model_name: str, dataset_name: str, output_dir: str = "./outputs/vlm_results/") -> None:
    """Save inference results to CSV file."""
    output_path = Path(output_dir) / f"{model_name}_{dataset_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, mode="w", encoding="utf-8", newline='') as f:
        if data:
            writer = csv.DictWriter(f, fieldnames=data[0].keys(), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
    
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="VLM Inference on VQA Datasets")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (file in models/)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (key in DATASET_CONFIGS)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--output-dir", type=str, 
                        default="./outputs/vlm_results/",
                        help="Output directory for results")
    parser.add_argument("--report-flops", action="store_true",
                        help="Measure and report FLOPs using PyTorch profiler")
    
    args = parser.parse_args()
    set_seed()
    
    # Load configuration and inference function
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{args.dataset}' not found in configuration")
    
    config = DATASET_CONFIGS[args.dataset]
    dataset_name = args.dataset
    inference_fn = load_inference_module(args.model)
    
    # Load and process data
    with open(config.qa_file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    # Load QAS cache for NARRATOR dataset
    qas_cache = None
    if dataset_name == "narrator_val":
        print("Loading NARRATOR QAS cache...")
        dataset_dir = str(Path(config.qa_file).parent)
        qas_cache = load_narrator_qas_cache(dataset_dir)
        print(f"Loaded QAS data for {len(qas_cache)} images")
    
    qids, img_ids, questions, gts, img_paths = process_batch_data(data, dataset_name, config, qas_cache)
    
    # Datasets that use most common answer for ground truth
    vqa_datasets = {"vqav2_restval", "textvqa_val", "vqav2_val"}
    
    total_flops: int = 0
    total_samples: int = 0

    for i in tqdm(range(0, len(questions), args.batch_size), desc="Processing batches"):
        batch_slice = slice(i, i + args.batch_size)
        batch_questions = questions[batch_slice]
        batch_images = img_paths[batch_slice]
        batch_gts = gts[batch_slice]
        
        batch_answers: Optional[List[str]] = None
        batch_flops: Optional[int] = None

        if args.report_flops and _PROFILER_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                profile_memory=False,
                with_modules=False,
            ) as prof:
                batch_answers = inference_fn(batch_questions, batch_images, config=config)
            torch.cuda.synchronize()

            # Sum FLOPs across all recorded operators (CUDA ops typically carry FLOPs)
            try:
                batch_flops = int(sum([e.flops for e in prof.key_averages() if hasattr(e, "flops") and e.flops is not None]))
            except Exception:
                batch_flops = None
        else:
            batch_answers = inference_fn(batch_questions, batch_images, config=config)
        
        print(f"Predictions: {batch_answers}")
        if dataset_name in vqa_datasets:
            gt_display = [most_common_answer(gt) for gt in batch_gts]
        else:
            gt_display = batch_gts
        print(f"Ground Truth: {gt_display}")
        
        # Compute per-sample FLOPs if available
        per_sample_flops: Optional[float] = None
        if args.report_flops:
            if not _PROFILER_AVAILABLE:
                print("[FLOPs] Torch profiler not available; skipping FLOPs measurement.")
            elif not torch.cuda.is_available():
                print("[FLOPs] CUDA not available; FLOPs metrics may be unavailable; skipping.")
            elif batch_flops is not None and len(batch_answers) > 0:
                per_sample_flops = float(batch_flops) / float(len(batch_answers))
                total_flops += batch_flops
                total_samples += len(batch_answers)
                print(f"[FLOPs] Batch total: {batch_flops:,} | Per-sample: {int(per_sample_flops):,}")

        # Update data with results
        for j, answer in enumerate(batch_answers):
            idx = i + j
            
            # For NARRATOR dataset, gt is already extracted from QAS cache
            # For other datasets, gt is in data[idx]["answer"]
            if dataset_name == "narrator_val":
                gt_value = gts[idx]
            else:
                gt_value = data[idx]["answer"]
            
            # For NARRATOR, use "id" field as question_id and add question text
            if dataset_name == "narrator_val":
                if "question_id" not in data[idx]:
                    data[idx]["question_id"] = data[idx]["id"]
                data[idx]["question"] = questions[idx]
            
            data[idx].update({
                "predict": answer,
                "gt": gt_value,
                "image_id": img_ids[idx]
            })
            if per_sample_flops is not None:
                data[idx]["flops"] = int(per_sample_flops)
            # Remove unnecessary fields
            data[idx].pop("image", None)
            data[idx].pop("answer", None)
            # Remove NARRATOR-specific metadata fields from output
            if dataset_name == "narrator_val":
                data[idx].pop("qas_file", None)
                data[idx].pop("qa_type", None)
                data[idx].pop("qa_index", None)
    
    save_results(data, args.model, dataset_name, args.output_dir)

    if args.report_flops and total_samples > 0:
        avg_per_sample = total_flops / total_samples if total_samples > 0 else 0
        print(f"[FLOPs] Total: {total_flops:,} | Samples: {total_samples} | Average per-sample: {int(avg_per_sample):,}")


if __name__ == "__main__":
    main()
