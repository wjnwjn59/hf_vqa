import argparse
import json
import re
import ast
import os
import glob
import csv
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_CONFIGS


ARTICLES = {'a', 'an', 'the'}
MANUAL_MAP = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
CONTRACTIONS = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hes": "he's", "im": "i'm", "ive": "i've", "isnt": "isn't", "itd": "it'd", "itll": "it'll", "lets": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", "shant": "shan't", "shed": "she'd", "shes": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "somebodyd": "somebody'd", "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd", "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed": "we'd", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "wholl": "who'll", "whos": "who's", "whove": "who've", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "yall": "y'all", "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've"}

def fix_list_format(item: object) -> object:
    if not isinstance(item, str):
        return item
    match = re.match(r"^\[(.*)\]$", item.strip())
    if not match:
        return item
    content = match.group(1)
    corrected = re.sub(r"(?<!['\w])(\w[^,]*?)(?!['\w])", r"'\1'", content)
    try:
        return ast.literal_eval(f"[{corrected}]")
    except Exception:
        return item


def parse_to_list(text: object) -> list[str] | None:
    if not isinstance(text, str):
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, list):
        return [str(x).strip(" '") for x in parsed]
    return None


def to_float(text: object) -> float | None:
    try:
        return float(str(text).strip().strip('%'))
    except ValueError:
        return None


def anls_score(pred: str, gts: list[str], threshold: float = 0.5) -> float:
    pred_norm = normalize_infovqa_answer(pred)
    gts_norm = [normalize_infovqa_answer(gt) for gt in gts]

    def levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        if len(a) == 0:
            return len(b)
        if len(b) == 0:
            return len(a)
        prev = list(range(len(b) + 1))
        curr = [0] * (len(b) + 1)
        for i, ca in enumerate(a):
            curr[0] = i + 1
            for j, cb in enumerate(b):
                cost = 0 if ca == cb else 1
                curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
            prev, curr = curr, prev
        return prev[len(b)]
    scores: list[float] = []
    for gt in gts_norm:
        dist = levenshtein(pred_norm, gt)
        length = max(len(pred_norm), len(gt))
        score = 1.0 - dist / length if length > 0 else 0.0
        scores.append(score)
    best = max(scores) if scores else 0.0
    return best if best >= threshold else 0.0


def evaluate_single_answer(target: str, prediction: str, max_relative_change: float = 0.05) -> float:
    t, p = target.strip().strip('%'), prediction.strip().strip('%')
    t_f, p_f = to_float(t), to_float(p)
    if t_f is not None and p_f is not None:
        if t_f == 0.0:
            return 1.0 if p_f == 0.0 else 0.0
        return 1.0 if abs(p_f - t_f) / abs(t_f) <= max_relative_change else 0.0
    return anls_score(p.lower(), [t.lower()])


def relaxed_correctness_chartqapro(
    target: str,
    prediction: str,
    max_relative_change: float = 0.05,
    year_flags: list[object] | None = None,
    always_use_exact_match: bool = False
) -> float:
    t_list = parse_to_list(str(fix_list_format(target))) or [str(target)]
    p_list = parse_to_list(str(prediction)) or [str(prediction)]
    n = len(t_list)
    if year_flags and len(year_flags) < n:
        year_flags = year_flags * n
    scores = []
    for idx in range(max(len(t_list), len(p_list))):
        if idx >= len(t_list) or idx >= len(p_list):
            scores.append(0.0)
            continue
        t_item, p_item = t_list[idx], p_list[idx]
        flag = year_flags[idx] if year_flags else 'NO'
        if (isinstance(flag, str) and flag.upper() == 'YES') or always_use_exact_match:
            scores.append(1.0 if str(t_item).strip().lower() == str(p_item).strip().lower() else 0.0)
        else:
            scores.append(evaluate_single_answer(t_item, p_item, max_relative_change))
    return sum(scores)/len(scores) if scores else 0.0


def evaluate_predictions_chartqapro(rows: list[dict]) -> dict[str, float]:
    gts = [x['gt'].strip(".\n") for x in rows]
    preds = [x['predict'].strip(".\n") for x in rows]
    splits = [x['question_type'] for x in rows]
    for x in rows:
        if x.get('year') is None:
            print(x)
    year_flags = [
        x['year'] if isinstance(x['year'], list) else
        ast.literal_eval(x['year']) if x['year'].startswith(
            '[') else [x['year']]
        for x in rows
    ]
    match_nums_per_split, match_nums = {}, []
    for gt, pred, split, year_flags_per_row in zip(gts, preds, splits, year_flags):
        if split == 'Conversational': continue
        match_nums_per_split.setdefault(split, [])
        always_use_exact_match = split in ['Fact Checking', 'Multi Choice']
        score = relaxed_correctness_chartqapro(gt, pred, year_flags=year_flags_per_row, always_use_exact_match=always_use_exact_match)
        match_nums_per_split[split].append(score)
        match_nums.append(score)
    final_numbers = {split: round(sum(vals)/len(vals)*100, 4) for split, vals in match_nums_per_split.items()}
    final_numbers['Overall'] = round(sum(match_nums)/len(match_nums)*100, 4)
    return final_numbers


def normalize_vqa_answer(ans: str) -> str:
    if len(ans) == 1:
        return ans.lower()
    punct = [';', r'/', '[', ']', '"',
             '}', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
    period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
    comma_strip = re.compile(r'(\d)(,)(\d)')

    out_text = ans
    for p in punct:
        if (p + ' ' in out_text or ' ' + p in out_text) or (re.search(comma_strip, out_text) is not None):
            out_text = out_text.replace(p, '')
        else:
            out_text = out_text.replace(p, ' ')
    out_text = period_strip.sub('', out_text)

    words = [
        CONTRACTIONS.get(word, word)
        for word in out_text.lower().split()
        if word not in ARTICLES
    ]
    words = [MANUAL_MAP.get(word, word) for word in words]
    return ' '.join(words).strip()


def relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
    def _to_float(text: object) -> float | None:
        try:
            text = str(text).strip()
            if text.endswith('%'):
                return float(text.rstrip('%')) / 100.0
            return float(text)
        except:
            return None
    pred_f, tgt_f = _to_float(prediction), _to_float(target)
    if pred_f is not None and tgt_f:
        return abs(pred_f - tgt_f) / abs(tgt_f) <= max_relative_change
    return str(prediction).lower().strip() == str(target).lower().strip()


def vqa_score(pred: str, gts: list[dict]) -> float:
    pred_norm = normalize_vqa_answer(pred)
    gts_norm = [normalize_vqa_answer(gt['answer']) for gt in gts]
    matches = [pred_norm == gt for gt in gts_norm]
    return min(1.0, sum(matches) / 3.0) if gts_norm else 0.0


def normalize_infovqa_answer(ans: str) -> str:
    ans = ans.lower().strip()
    ans = re.sub(r'\s+', ' ', ans)
    ans = re.sub(r'[\.,;:!?"\'\[\](){}]', '', ans)
    for k, v in MANUAL_MAP.items():
        ans = re.sub(r'\b' + re.escape(k) + r'\b', v, ans)
    return ans


def parse_gt_field(raw_gt: str, metric: str):
    """
    Convert the CSV 'gt' column back into the original Python object.
    For vqav2_val, raw_gt is something like "[{'answer': 'yes'}, ...]".
    For others, it's just a single string.
    """
    if metric in ['vqa_score', 'chartqapro_score']:
        return ast.literal_eval(raw_gt)

    else:
        try:
            val = ast.literal_eval(raw_gt)
        except Exception:
            val = raw_gt
        return val


def calculate_metric_scores(rows, metric):
    """
    Calculate scores for each row based on the specified metric.

    Args:
        rows: List of dictionary rows from CSV
        metric: Metric to use for evaluation

    Returns:
        List of scores for each row
    """
    scores = []
    for row in rows:
        pred: str | list[str] = row['predict']
        if isinstance(pred, str):
            pred = pred.replace('the answer is', '').replace('The answer is', '').rstrip('.').strip()
        else:
            pred = pred[0].replace('the answer is', '').replace('The answer is', '').rstrip('.').strip()
        gts = parse_gt_field(row['gt'], metric)

        if metric == 'vqa_score':
            if not isinstance(gts, list):
                raise TypeError("gts must be a list of dictionaries for vqa_score")
            score = vqa_score(pred, gts)
        elif metric == 'anls':
            gts_list = [str(gt) for gt in (gts if isinstance(gts, list) else [gts])]
            score = anls_score(pred, gts_list, threshold=0.5)
        elif metric == 'relaxed_accuracy':
            ann_list = [str(ann) for ann in (gts if isinstance(gts, list) else [gts])]
            score = 1.0 if any(relaxed_correctness(ann, pred) for ann in ann_list) else 0.0
        else:
            raise ValueError(f"Unknown metric {metric}")
        scores.append(score)
    return scores


def evaluate_folder(folder_path: str) -> None:
    """
    Evaluate model performance on different datasets from CSV files in a folder.

    Args:
        folder_path: Path to folder containing CSV result files
    """
    output_json_path = os.path.join(folder_path, "models_scores.json")
    results = {}
    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
        base = os.path.basename(csv_file).lower()
        model_name, dataset = base[:-4].split('_', 1)
        model_results = results.setdefault(model_name, {})
        existing_scores = model_results.get(dataset, {})
        main_score = existing_scores.get('main_score')

        with open(csv_file, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))

        if main_score is None:
            metric = DATASET_CONFIGS[dataset].metric
            if metric == 'chartqapro':
                main_score = evaluate_predictions_chartqapro(rows)
            else:
                scores = calculate_metric_scores(rows, metric)
                main_score = sum(scores) / len(scores) * 100 if scores else 0.0

        final_scores = {
            "main_score": round(main_score, 4) if isinstance(main_score, float) else main_score,
        }

        results[model_name][dataset] = final_scores
        print(f"  -> Score of {base}: {final_scores}")

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\n✅ All evaluations complete. Results saved to {output_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate all models on all datasets in a folder.")
    parser.add_argument('--folder', type=str, required=True, help="Folder containing CSV result files.")
    args = parser.parse_args()
    evaluate_folder(args.folder)