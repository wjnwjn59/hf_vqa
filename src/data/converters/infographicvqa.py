import os
import json
import argparse
import shutil

from pathlib import Path

SPLIT_FILES = {
    "train": "infographicsVQA_train_v1.0.json",
    "val":   "infographicsVQA_val_v1.0_withQT.json",
    "test":  "infographicsVQA_test_v1.0.json",
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_raw_questions(src_root: Path, split: str):
    qa_dir = src_root / "question_answer"
    src_file = qa_dir / SPLIT_FILES[split]
    with open(src_file, "r", encoding="utf-8") as f:
        return json.load(f)

def main(src: str, dst: str, copy_images: bool, copy_ocr: bool):
    src_root, dst_root = Path(src), Path(dst)
    img_src = src_root / "images"
    ocr_src = src_root / "ocr"

    ann_dir = dst_root / "annotations"
    img_dst = dst_root / "images"
    ocr_dst = dst_root / "ocr"

    ensure_dir(ann_dir); ensure_dir(img_dst)
    if copy_ocr and ocr_src.exists():
        ensure_dir(ocr_dst)

    for split in ["train", "val", "test"]:
        raw = load_raw_questions(src_root, split)
        out = ann_dir / f"{split}.jsonl"
        cnt = 0
        with open(out, "w", encoding="utf-8") as w:
            for ex in raw:
                img_name = ex["image_local_name"]
                src_img = img_src / img_name
                dst_img = img_dst / img_name

                if not dst_img.exists():
                    if copy_images:
                        shutil.copy2(src_img, dst_img)
                    else:
                        try:
                            os.symlink(src_img, dst_img)
                        except FileExistsError:
                            pass

                ocr_file = ex.get("ocr_output_file")
                if ocr_file and copy_ocr and (ocr_src / ocr_file).exists():
                    dst_ocr = ocr_dst / ocr_file
                    if not dst_ocr.exists():
                        shutil.copy2(ocr_src / ocr_file, dst_ocr)

                item = {
                    "id": f"infovqa_{split}_{ex['questionId']}",
                    "image": f"images/{img_name}",
                    "question": ex["question"],
                    "answers": ex.get("answers", []),
                    "dataset": "infographicvqa",
                    "split": split,
                    "meta": {
                        "image_url": ex.get("image_url"),
                        "ocr_file": f"ocr/{ocr_file}" if ocr_file else None
                    },
                    "eval": { "match": "anls" }
                }
                w.write(json.dumps(item, ensure_ascii=False) + "\n")
                cnt += 1
        print(f"[{split}] wrote {cnt} → {out}")

    manifest = {
        "name": "infographicvqa",
        "version": "v1.0",
        "splits": {
            s: sum(1 for _ in open(ann_dir / f"{s}.jsonl", "r", encoding="utf-8"))
            for s in ["train", "val", "test"]
        }
    }
    with open(dst_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote manifest.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--copy-images", action="store_true")
    ap.add_argument("--copy-ocr", action="store_true")
    args = ap.parse_args()
    main(args.src, args.dst, copy_images=args.copy_images, copy_ocr=args.copy_ocr)