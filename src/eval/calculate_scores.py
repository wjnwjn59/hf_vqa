import os
import json
import argparse
import torch
import Levenshtein
import regex as re
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.inference.models.utils import extract_clean_model_name, extract_clean_filename

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "src/inference/results"

llms = [
    "/mnt/dataset1/pretrained_fm/Tower-Babel_Babel-9B-Chat",
    "/mnt/dataset1/pretrained_fm/Qwen_Qwen3-8B",
    "/mnt/dataset1/pretrained_fm/internlm_internlm3-8b-instruct",
]

PROMPT_TEMPLATE = """
You are an AI assistant specialized in evaluating predictions against ground truth answers.
**Rules:** - Each sample has a maximum score of 1 point; if GT has multiple sub-answers, score = (matched sub-answers) / (total sub-answers in GT).
- Order does not matter; different phrasings with the same meaning are considered matches.
- Do NOT add explanations, only return in this format:
Score: <matched sub-answers>/<total sub-answers>

**Examples:**
Question: What types of vehicles are in the image?
GT: motorcycle, bicycle, car
Predict: automobile, motorbike
Score: 2/3
Explain: Because 2/3 answers match GT ("motorcycle" is synonymous with "motorbike", "car" is synonymous with "automobile").

Question: Which years did the company have profits over 500 million?
GT: year 2010, year 2015
Predict: 2010 and 2015
Score: 2/2
Explain: Because both years are in GT.

Question: How many people are in the picture?
GT: 5
Predict: there are 5 people in the picture
Score: 1/1
Explain: The answer matches GT, just slightly longer.

Question: What is the predicted temperature on 21/2/2020 in degrees C?
GT: 30 degrees
Predict: 27 degrees
Score: 0/1
Explain: The answer does not match GT.

**Begin evaluation:**
Question: {question}
GT: {gt}
Predict: {predict}
Score:""".strip()


def parse_score(gen: str):
    """Parses LLM output to extract score as float."""
    text = gen.split("Score:")[-1].strip()
    m = re.match(r"^(\d+)\s*/\s*(\d+)", text)
    if not m:
        return 0.0
    num, den = map(int, m.groups())
    return num / den if den else 0.0


def compute_anls(gt, predict, threshold: float = 0.5):
    """
    Computes ANLS (Average Normalized Levenshtein Similarity) score.
    If gt is a list, returns max score across all ground truths.
    """
    p = str(predict).replace('"', '').rstrip('.').lower()
    
    # Handle list of ground truths (InfographicVQA format)
    if isinstance(gt, list):
        if not gt:
            return 0.0
        scores = [Levenshtein.ratio(p, str(g).lower()) for g in gt]
        scores = [s if s >= threshold else 0.0 for s in scores]
        return max(scores)
    
    # Handle single ground truth string
    score = Levenshtein.ratio(p, str(gt).lower())
    return score if score >= threshold else 0.0


def compute_llm_scores_batch(questions, gts, prs, tokenizer, model):
    """Computes LLM-based scores for a batch of predictions."""
    # Format GT: if list, use first answer for LLM prompt (representative)
    gts_formatted = [gt[0] if isinstance(gt, list) and gt else str(gt) for gt in gts]
    
    prompts = [PROMPT_TEMPLATE.format(question=q, gt=gt, predict=pr) 
               for q, gt, pr in zip(questions, gts_formatted, prs)]
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, 
                      truncation=True, max_length=600)
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out_ids = model.generate(**inputs, max_new_tokens=5, 
                                eos_token_id=tokenizer.eos_token_id)
    
    return [parse_score(tokenizer.decode(out_ids[i], skip_special_tokens=True).strip()) 
            for i in range(len(prompts))]


def analyze_by_categories(preds):
    """Analyzes predictions by categories and computes scores."""
    category_scores = {
        "Overall": {"total_anls": 0.0, "total_accuracy": 0.0, "total_llm": 0.0, "count": 0},
        "Answer type": defaultdict(lambda: {"total_anls": 0.0, "total_accuracy": 0.0, "total_llm": 0.0, "count": 0}),
        "Element": defaultdict(lambda: {"total_anls": 0.0, "total_accuracy": 0.0, "total_llm": 0.0, "count": 0}),
        "Operation": defaultdict(lambda: {"total_anls": 0.0, "total_accuracy": 0.0, "total_llm": 0.0, "count": 0}),
    }
    
    # Mapping from original keys to category names
    key_mapping = {
        "answer_source": "Answer type",
        "answer_type": "Answer type",
        "element": "Element",
        "evidence": "Element",
        "operation": "Operation",
        "operation/reasoning": "Operation"
    }
    
    for item in preds:
        gt_answer = item['answer']
        pred_answer = str(item.get('predict', ''))
        
        pr_clean = pred_answer.lower().strip().rstrip('.').replace('"', '').rstrip('>').lstrip('<')
        
        # Compute ANLS (handles both list and string GT)
        anls_score = compute_anls(gt_answer, pr_clean)
        
        # Compute accuracy (if GT is list, check if pred matches any)
        if isinstance(gt_answer, list):
            gt_answers_clean = [str(g).lower().strip() for g in gt_answer]
            accuracy_score = int(any(pr_clean == g for g in gt_answers_clean))
        else:
            gt_clean = str(gt_answer).lower().strip()
            accuracy_score = int(gt_clean == pr_clean)
        
        llm_score = float(item.get('llm_score', 0.0))
        
        item['anls'] = anls_score
        item['accuracy'] = accuracy_score
        
        overall = category_scores["Overall"]
        overall["total_anls"] += anls_score
        overall["total_accuracy"] += accuracy_score
        overall["total_llm"] += llm_score
        overall["count"] += 1
        
        # Process categories with flexible key mapping
        for category_key_orig, category_key_mapped in key_mapping.items():
            value = item.get(category_key_orig)
            if value is None:
                continue
                
            values_to_process = (value if isinstance(value, list) and value else
                               [value] if value is not None else ["N/A"])
            
            for sub_category in values_to_process:
                normalized_key = str(sub_category).lower()
                stats = category_scores[category_key_mapped][normalized_key]
                stats["total_anls"] += anls_score
                stats["total_accuracy"] += accuracy_score
                stats["total_llm"] += llm_score
                stats["count"] += 1
    
    return category_scores


def calculate_averages(stats):
    """Calculates average scores as percentages."""
    count = stats['count']
    if count == 0:
        return {'accuracy': 0.0, 'anls': 0.0, 'llm_score': 0.0, 'count': 0}
    return {
        'accuracy': round((stats['total_accuracy'] / count) * 100, 2),
        'anls': round((stats['total_anls'] / count) * 100, 2),
        'llm_score': round((stats['total_llm'] / count) * 100, 2),
    }


def create_detailed_report(category_scores):
    """Creates detailed report with scores for all categories."""
    report = {"Overall": calculate_averages(category_scores["Overall"])}
    for category_key, sub_categories in category_scores.items():
        if category_key == "Overall":
            continue
        report[category_key] = {
            sub_category: calculate_averages(stats)
            for sub_category, stats in sorted(sub_categories.items())
        }
    return report


def save_analysis_to_txt(detailed_analysis: dict, filename: str, metric: str):
    """Saves detailed analysis to tab-separated TXT file."""
    main_categories_order = ["Answer type", "Element", "Operation"]
    excluded_keys = {'none', '[]', 'n/a'}
    
    all_sub_categories = defaultdict(set)
    for analysis_data in detailed_analysis.values():
        for category in main_categories_order:
            if category in analysis_data:
                all_sub_categories[category].update(analysis_data[category].keys())
    
    final_column_structure = {}
    header = ["Model", "Overall"]
    for category in main_categories_order:
        valid_sub_cats = sorted([
            key for key in all_sub_categories[category] 
            if str(key).lower() not in excluded_keys
        ])
        final_column_structure[category] = valid_sub_cats
        header.extend([sub.replace("-", " ").replace("_", " ").title() 
                      for sub in valid_sub_cats])
    
    all_rows = []
    for model_name, analysis_data in sorted(detailed_analysis.items()):
        row = [model_name]
        row.append(analysis_data.get("Overall", {}).get(metric, 0))
        
        for category in main_categories_order:
            for sub_category in final_column_structure[category]:
                score = analysis_data.get(category, {}).get(sub_category, {}).get(metric, 0)
                row.append(score)
        all_rows.append(row)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\t".join(header) + "\n")
        for row in all_rows:
            f.write("\t".join(map(str, row)) + "\n")


def load_all_prediction_files():
    """Loads all prediction JSON files from results directory."""
    predict_files = sorted([
        f for f in os.listdir(RESULTS_DIR) 
        if f.endswith(".json") and 'scores' not in f and 'analysis' not in f
    ])
    
    file_data = {}
    for fname in predict_files:
        file_path = os.path.join(RESULTS_DIR, fname)
        with open(file_path, "r", encoding="utf-8") as f:
            file_data[fname] = json.load(f)
    
    return file_data


def save_file_data(fname, data):
    """Saves prediction data back to JSON file."""
    file_path = os.path.join(RESULTS_DIR, fname)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_final_metrics(file_data):
    """Computes final metrics for all prediction files."""
    results = {}
    detailed_analysis = {}
    
    for fname, preds in file_data.items():
        key = extract_clean_filename(fname)
        
        # Compute average LLM score from multiple LLM judges
        llm_fields = [extract_clean_model_name(model_name) + "_score" for model_name in llms]
        for entry in preds:
            valid_llm_scores = [entry[f] for f in llm_fields if f in entry]
            entry["llm_score"] = (sum(valid_llm_scores) / len(valid_llm_scores) 
                                if valid_llm_scores else 0.0)
        
        category_scores = analyze_by_categories(preds)
        detailed_analysis[key] = create_detailed_report(category_scores)
        
        overall_metrics = detailed_analysis[key]['Overall']
        results[key] = {
            "accuracy": overall_metrics['accuracy'],
            "anls": overall_metrics['anls'],
            "llm_score": overall_metrics['llm_score']
        }
        
        save_file_data(fname, preds)
    
    return results, detailed_analysis


def run_llm_scoring_for_files(file_data):
    """Runs LLM scoring for all files that don't have scores yet."""
    for model_name in llms:
        clean_name = extract_clean_model_name(model_name)
        field = f"{clean_name}_score"
        
        # Find files that need scoring
        files_to_process = {
            fname: data for fname, data in file_data.items()
            if any(field not in p for p in data)
        }
        
        if not files_to_process:
            print(f"✅ {clean_name} scores already exist, skipping...")
            continue
        
        print(f"🔄 Loading {clean_name} model for {len(files_to_process)} files...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16, padding_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto", low_cpu_mem_usage=True
        ).eval()
        
        for fname, preds in files_to_process.items():
            clean_fname = extract_clean_filename(fname)
            print(f"   ⏳ Computing {clean_name} scores for {clean_fname}...")
            
            batch_size = 32
            all_scores = []
            
            for start in tqdm(range(0, len(preds), batch_size), desc=f"      {clean_name}"):
                end = min(start + batch_size, len(preds))
                batch_data = preds[start:end]
                
                questions = [item.get("question", "") for item in batch_data]
                gts = [item.get("answer", "") for item in batch_data]
                prs = [str(item.get("predict", "")).lower().rstrip('.') for item in batch_data]
                
                scores = compute_llm_scores_batch(questions, gts, prs, tokenizer, model)
                all_scores.extend(scores)
            
            for i, score in enumerate(all_scores):
                preds[i][field] = score
            
            save_file_data(fname, preds)
            print(f"   💾 Saved {clean_name} scores for {clean_fname}")
        
        del model, tokenizer
        torch.cuda.empty_cache()
        print(f"✅ Completed {clean_name} scoring")


def main(llm_scores: bool = False):
    """Main function to calculate scores for all prediction files."""
    print("🚀 Starting score calculation...")
    
    file_data = load_all_prediction_files()
    if not file_data:
        print("No prediction files found. Exiting...")
        return
    
    print(f"📁 Loaded {len(file_data)} prediction files")
    
    if llm_scores:
        print("🧠 LLM scoring: ENABLED")
        run_llm_scoring_for_files(file_data)
    else:
        print("🧠 LLM scoring: DISABLED")
    
    print("📈 Computing final metrics...")
    results, detailed_analysis = compute_final_metrics(file_data)
    
    print("💾 Saving results...")
    final_scores_path = os.path.join(RESULTS_DIR, "final_scores.json")
    detailed_analysis_path = os.path.join(RESULTS_DIR, "detailed_analysis.json")
    
    with open(final_scores_path, "w", encoding="utf-8") as fw:
        json.dump(results, fw, indent=2, ensure_ascii=False)
    
    with open(detailed_analysis_path, "w", encoding="utf-8") as fw:
        json.dump(detailed_analysis, fw, indent=2, ensure_ascii=False)
    
    print("✅ Final Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    print("\n💾 Saving TXT reports...")
    acc_path = os.path.join(RESULTS_DIR, "scores_accuracy.txt")
    anls_path = os.path.join(RESULTS_DIR, "scores_anls.txt")
    llm_path = os.path.join(RESULTS_DIR, "scores_llm.txt")
    
    save_analysis_to_txt(detailed_analysis, acc_path, metric='accuracy')
    save_analysis_to_txt(detailed_analysis, anls_path, metric='anls')
    save_analysis_to_txt(detailed_analysis, llm_path, metric='llm_score')
    
    print(f"✅ Accuracy scores saved to: {acc_path}")
    print(f"✅ ANLS scores saved to: {anls_path}")
    print(f"✅ LLM scores saved to: {llm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate VQA scores with multiple metrics")
    parser.add_argument(
        "--llm-scores", dest="llm_scores", action="store_true",
        help="Enable LLM-based scoring (slower but more accurate)"
    )
    parser.add_argument(
        "--no-llm-scores", dest="llm_scores", action="store_false",
        help="Disable LLM-based scoring (faster, uses only ANLS and exact match)"
    )
    parser.set_defaults(llm_scores=False)
    args = parser.parse_args()
    
    main(llm_scores=args.llm_scores)

