import os
import json
import Levenshtein
from collections import defaultdict
from eval.utils import extract_clean_filename

# RESULTS_DIR = "eval/results"
RESULTS_DIR = "training/LLaMA-Factory/experiments"


def is_numerical(text):
    """Check if text represents a number."""
    if not isinstance(text, str):
        text = str(text)
    cleaned = text.strip().replace(',', '').replace('%', '').replace('$', '')
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def parse_number(text):
    """Parse text to number, handling common formats."""
    if not isinstance(text, str):
        text = str(text)
    cleaned = text.strip().replace(',', '').replace('%', '').replace('$', '')
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_multiple_choice(answer):
    """Check if answer is multiple choice (Yes/No type)."""
    if not isinstance(answer, str):
        answer = str(answer)
    answer_lower = answer.strip().lower()
    # Common multiple choice answers
    mc_answers = {'yes', 'no', 'true', 'false', 'a', 'b', 'c', 'd', 'e'}
    return answer_lower in mc_answers


def compute_numerical_accuracy(gt, predict, tolerance: float = 0.05):
    """
    Computes numerical accuracy with relative tolerance (5% for ChartGalaxy).
    Returns 1.0 if within tolerance, 0.0 otherwise.
    """
    gt_num = parse_number(gt)
    pred_num = parse_number(predict)
    
    if gt_num is None or pred_num is None:
        return 0.0
    
    # Exact match
    if gt_num == pred_num:
        return 1.0
    
    # Handle zero case
    if gt_num == 0:
        return 1.0 if pred_num == 0 else 0.0
    
    # Relative tolerance (5% = 0.05)
    # Use small epsilon (1e-9) to handle floating point comparison
    relative_diff = abs(gt_num - pred_num) / abs(gt_num)
    return 1.0 if relative_diff <= tolerance + 1e-9 else 0.0


def compute_anls(gt, predict, threshold: float = 0.5):
    """
    Computes ANLS (Average Normalized Levenshtein Similarity) score.
    If gt is a list, returns max score across all ground truths.
    Used for: InfographicVQA (all samples) and ChartGalaxy (text answers only).
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


def compute_chartgalaxy_accuracy(gt, predict, answer_type=None):
    """
    Computes accuracy for ChartGalaxy dataset using appropriate metric.
    
    ChartGalaxy evaluation rules:
    - Numerical: Relaxed accuracy with 5% tolerance
    - Multiple choice (Yes/No): Exact match
    - Text: ANLS with 0.5 threshold
    """
    pred_clean = str(predict).strip().lower().replace('"', '').rstrip('.').rstrip('>').lstrip('<')
    gt_str = str(gt).strip()
    
    # 1. Numerical answers (calculation type or numerical format)
    if is_numerical(gt_str):
        return compute_numerical_accuracy(gt, predict, tolerance=0.05)
    
    # 2. Multiple choice (Yes/No, True/False, etc.)
    if is_multiple_choice(gt_str):
        gt_clean = gt_str.lower().strip()
        return 1.0 if pred_clean == gt_clean else 0.0
    
    # 3. Text answers (use ANLS)
    return compute_anls(gt, predict, threshold=0.5)


def analyze_by_categories(preds):
    """Analyzes predictions by categories and computes scores."""
    category_scores = {
        "Overall": {"total_anls": 0.0, "total_accuracy": 0.0, "count": 0},
        "Answer type": defaultdict(lambda: {"total_anls": 0.0, "total_accuracy": 0.0, "count": 0}),
        "Element": defaultdict(lambda: {"total_anls": 0.0, "total_accuracy": 0.0, "count": 0}),
        "Operation": defaultdict(lambda: {"total_anls": 0.0, "total_accuracy": 0.0, "count": 0}),
    }
    
    # InfographicVQA format mapping
    infographic_mapping = {
        "answer_source": "Answer type",
        "answer_type": "Answer type",
        "element": "Element",
        "evidence": "Element",
        "operation": "Operation",
        "operation/reasoning": "Operation"
    }
    
    # ChartGalaxy metadata mapping
    chartgalaxy_metadata_mapping = {
        "type": "Answer type",
        "category": "Element",
        "subcategory": "Operation"
    }
    
    for item in preds:
        gt_answer = item['answer']
        pred_answer = str(item.get('predict', ''))
        if pred_answer.count("ANSWER: "):
            pred_answer = pred_answer.split("ANSWER: ")[-1].strip().lower()
            
        
        pr_clean = pred_answer.lower().strip().rstrip('.').replace('"', '').rstrip('>').lstrip('<')
        
        # Detect dataset type
        is_chartgalaxy = 'metadata' in item and isinstance(item.get('metadata'), dict)
        
        if is_chartgalaxy:
            answer_type = item.get('metadata', {}).get('type', None)
            accuracy_score = compute_chartgalaxy_accuracy(gt_answer, pred_answer, answer_type)
            anls_score = accuracy_score
        else:
            anls_score = compute_anls(gt_answer, pr_clean)
            if isinstance(gt_answer, list):
                gt_answers_clean = [str(g).lower().strip() for g in gt_answer]
                accuracy_score = int(any(pr_clean == g for g in gt_answers_clean))
            else:
                gt_clean = str(gt_answer).lower().strip()
                accuracy_score = int(gt_clean == pr_clean)
        
        item['anls'] = anls_score
        item['accuracy'] = accuracy_score
        
        overall = category_scores["Overall"]
        overall["total_anls"] += anls_score
        overall["total_accuracy"] += accuracy_score
        overall["count"] += 1
        
        # Process InfographicVQA categories
        for category_key_orig, category_key_mapped in infographic_mapping.items():
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
                stats["count"] += 1
        
        # Process ChartGalaxy metadata (if present)
        if 'metadata' in item and isinstance(item['metadata'], dict):
            for meta_key, category_key_mapped in chartgalaxy_metadata_mapping.items():
                value = item['metadata'].get(meta_key)
                if value is None:
                    continue
                
                normalized_key = str(value).lower()
                stats = category_scores[category_key_mapped][normalized_key]
                stats["total_anls"] += anls_score
                stats["total_accuracy"] += accuracy_score
                stats["count"] += 1
    
    return category_scores


def calculate_averages(stats):
    """Calculates average scores as percentages."""
    count = stats['count']
    if count == 0:
        return {'accuracy': 0.0, 'anls': 0.0}
    return {
        'accuracy': round((stats['total_accuracy'] / count) * 100, 2),
        'anls': round((stats['total_anls'] / count) * 100, 2),
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
    """Computes final metrics for all prediction files, grouped by dataset."""
    results_by_dataset = defaultdict(dict)
    analysis_by_dataset = defaultdict(dict)
    
    for fname, preds in file_data.items():
        model_name, dataset_name = extract_clean_filename(fname)
        
        category_scores = analyze_by_categories(preds)
        detailed_report = create_detailed_report(category_scores)
        
        analysis_by_dataset[dataset_name][model_name] = detailed_report
        
        overall_metrics = detailed_report['Overall']
        results_by_dataset[dataset_name][model_name] = {
            "accuracy": overall_metrics['accuracy'],
            "anls": overall_metrics['anls']
        }
        
        save_file_data(fname, preds)
    
    return results_by_dataset, analysis_by_dataset


def main():
    """Main function to calculate scores for all prediction files."""
    print("🚀 Starting score calculation...")
    
    file_data = load_all_prediction_files()
    if not file_data:
        print("No prediction files found. Exiting...")
        return
    
    print(f"📁 Loaded {len(file_data)} prediction files")
    print("📈 Computing final metrics...")
    results_by_dataset, analysis_by_dataset = compute_final_metrics(file_data)
    
    print("💾 Saving results by dataset...")
    
    # Save results for each dataset separately
    for dataset_name in results_by_dataset.keys():
        print(f"\n📊 Dataset: {dataset_name}")
        
        # Save JSON results
        final_scores_path = os.path.join(RESULTS_DIR, f"final_scores_{dataset_name}.json")
        detailed_analysis_path = os.path.join(RESULTS_DIR, f"detailed_analysis_{dataset_name}.json")
        
        with open(final_scores_path, "w", encoding="utf-8") as fw:
            json.dump(results_by_dataset[dataset_name], fw, indent=2, ensure_ascii=False)
        
        with open(detailed_analysis_path, "w", encoding="utf-8") as fw:
            json.dump(analysis_by_dataset[dataset_name], fw, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Final scores: {final_scores_path}")
        print(f"  ✅ Detailed analysis: {detailed_analysis_path}")
        
        # Save TXT reports
        acc_path = os.path.join(RESULTS_DIR, f"scores_accuracy_{dataset_name}.txt")
        anls_path = os.path.join(RESULTS_DIR, f"scores_anls_{dataset_name}.txt")
        
        save_analysis_to_txt(analysis_by_dataset[dataset_name], acc_path, metric='accuracy')
        save_analysis_to_txt(analysis_by_dataset[dataset_name], anls_path, metric='anls')
        
        print(f"  ✅ Accuracy: {acc_path}")
        print(f"  ✅ ANLS: {anls_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SUMMARY BY DATASET")
    print("="*60)
    for dataset_name, results in results_by_dataset.items():
        print(f"\n{dataset_name.upper()}:")
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
