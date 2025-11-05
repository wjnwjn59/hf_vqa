#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

def load_json_list(p: Path) -> List[Dict[str, Any]]:
    """Load a JSON file that should contain a list of dicts.
    If it contains a single dict, wrap it as a list. Gracefully handles BOMs."""
    with p.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{p} does not contain a list or dict JSON payload.")
    return data

def count_markers(text: str) -> Dict[str, int]:
    """Count occurrences of (text) and (figure) markers (case-insensitive)."""
    s = (text or "").lower()
    return {
        "text_markers": s.count("(text)"),
        "figure_markers": s.count("(figure)"),
    }

def main(root: Path, failed_dir: Path):
    json_paths = sorted([p for p in root.glob("*.json") if p.is_file()])
    if not json_paths:
        print(f"No .json files found under: {root.resolve()}")
        return

    # Global aggregates
    g_total = 0
    g_kw_equal = g_kw_mismatch = g_kw_missing = 0
    g_text_markers = g_figure_markers = 0
    g_zero_figure_cases = []  # (filename, infographic_id, title)

    # Collect failed samples across all files.
    # We keep a 'seen' set keyed by a stable serialization to avoid duplicates.
    failed_samples: List[Dict[str, Any]] = []
    failed_seen = set()

    def _maybe_add_failed(sample: Dict[str, Any]):
        key = json.dumps(sample, ensure_ascii=False, sort_keys=True)
        if key not in failed_seen:
            failed_seen.add(key)
            failed_samples.append(sample)

    print("=" * 80)
    print(f"Scanning directory: {root.resolve()}")
    print("=" * 80)

    for jp in json_paths:
        try:
            samples = load_json_list(jp)
        except Exception as e:
            print(f"[SKIP] {jp.name}: failed to load JSON ({e})")
            continue

        total = len(samples)
        kw_equal = kw_mismatch = kw_missing = 0
        text_markers_sum = figure_markers_sum = 0
        zero_figure_cases = []  # list of (infographic_id, title)

        for s in samples:
            # ------- keywords_found vs keywords_total -------
            kf = s.get("keywords_found")
            kt = s.get("keywords_total")
            if kf is None or kt is None:
                kw_missing += 1
            elif kf == kt:
                kw_equal += 1
            else:
                kw_mismatch += 1
                # collect the entire dict for failed.json
                _maybe_add_failed(s)

            # ------- generated_infographic markers -------
            markers = count_markers(s.get("generated_infographic", ""))
            text_markers_sum += markers["text_markers"]
            figure_markers_sum += markers["figure_markers"]

            if markers["figure_markers"] == 0:
                zero_figure_cases.append((s.get("infographic_id"), s.get("title")))
                # ALSO include zero-figure samples in failed.json
                _maybe_add_failed(s)

        # File-level printout
        print(f"\n--- {jp.name} ---")
        print(f"Samples: {total}")
        print(f"keywords_found == keywords_total: {kw_equal}")
        print(f"keywords_found != keywords_total (MISMATCH): {kw_mismatch}")
        print(f"keywords_found/total MISSING: {kw_missing}")
        print(f"Total '(text)' markers: {text_markers_sum}  "
              f"(avg {text_markers_sum/total:.2f} per sample)")
        print(f"Total '(figure)' markers: {figure_markers_sum}  "
              f"(avg {figure_markers_sum/total:.2f} per sample)")
        print(f"Samples with ZERO '(figure)': {len(zero_figure_cases)}")

        if zero_figure_cases:
            print("  -> Cases with no '(figure)':")
            for inf_id, title in zero_figure_cases:
                print(f"     - id={inf_id}  title={title}")

        # Update globals
        g_total += total
        g_kw_equal += kw_equal
        g_kw_mismatch += kw_mismatch
        g_kw_missing += kw_missing
        g_text_markers += text_markers_sum
        g_figure_markers += figure_markers_sum
        g_zero_figure_cases.extend([(jp.name, *c) for c in zero_figure_cases])

    # Global summary
    print("\n" + "=" * 80)
    print("GLOBAL SUMMARY")
    print("=" * 80)
    print(f"Total samples: {g_total}")
    print(f"keywords_found == keywords_total: {g_kw_equal} "
          f"({(g_kw_equal/g_total):.3f} of samples)")
    print(f"keywords_found != keywords_total (MISMATCH): {g_kw_mismatch} "
          f"({(g_kw_mismatch/g_total):.3f} of samples)")
    print(f"keywords_found/total MISSING: {g_kw_missing} "
          f"({(g_kw_missing/g_total):.3f} of samples)")
    print(f"Total '(text)' markers: {g_text_markers}  "
          f"(avg {g_text_markers/g_total:.2f} per sample)")
    print(f"Total '(figure)' markers: {g_figure_markers}  "
          f"(avg {g_figure_markers/g_total:.2f} per sample)")
    print(f"Samples with ZERO '(figure)': {len(g_zero_figure_cases)} "
          f"({len(g_zero_figure_cases)/g_total:.3f} of samples)")

    if g_zero_figure_cases:
        print("\nFiles & samples with no '(figure)':")
        for fname, inf_id, title in g_zero_figure_cases:
            print(f"  {fname}: id={inf_id}  title={title}")

    # Write failed.json if there are any failures (mismatch or zero-figure)
    if failed_samples:
        failed_dir.mkdir(parents=True, exist_ok=True)
        out_path = failed_dir / "failed.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(failed_samples)} failed samples "
              f"(keyword mismatches and/or zero '(figure)') to: {out_path.resolve()}")
    else:
        print("\nNo failed samples to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-file and global stats for infographic JSONs."
    )
    parser.add_argument(
        "dir",
        nargs="?",
        default="create_data/output/final_infographic",
        help="Directory containing .json files (default: create_data/output/final_infographic)",
    )
    parser.add_argument(
        "--failed-dir",
        default="create_data/output/failed",
        help="Output directory for failed cases (failed.json). Default: create_data/output/failed",
    )
    args = parser.parse_args()
    main(Path(args.dir), Path(args.failed_dir))
