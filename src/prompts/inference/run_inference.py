import os
import json
import argparse
from tqdm import tqdm
from src.inference.utils import set_seed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODELS = {
    "internvl": "src.inference.models.internvl.InternVLModel",
    "llava": "src.inference.models.llava.LLaVAModel",
    "molmo": "src.inference.models.molmo.MolmoModel",
    "qwenvl": "src.inference.models.qwenvl.QwenVLModel",
    "videollama": "src.inference.models.videollama.VideoLLAMAModel",
    "phi": "src.inference.models.phi.PhiModel",
    "ovis": "src.inference.models.ovis.OvisModel",
    "minicpm": "src.inference.models.minicpm.MiniCPMModel",
}


def import_model_class(model_key: str):
    """Dynamically imports model class."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
    
    module_path, class_name = MODELS[model_key].rsplit('.', 1)
    
    try:
        print(f"📦 Importing {class_name}...")
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        print(f"❌ Failed to import {class_name}: {e}")
        raise
    except AttributeError as e:
        print(f"❌ Class {class_name} not found: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Run VLM inference")
    parser.add_argument("model", type=str, choices=MODELS.keys(), help="Model to run")
    parser.add_argument("--image_folder", type=str, default="/mnt/VLAI_data/InfographicVQA/images")
    parser.add_argument("--data_path", type=str, default="/mnt/VLAI_data/InfographicVQA/infographicvqa_val.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="src/inference/results")
    args = parser.parse_args()

    set_seed(args.seed)
    
    print(f"🚀 Initializing {args.model} model...")
    try:
        ModelClass = import_model_class(args.model)
        model = ModelClass()
        clean_model_name = model.model_name
        print(f"✅ Loaded {clean_model_name}")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return 1
    
    
    print(f"📂 Loading data from {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Auto-detect dataset type and set defaults
    if data and 'images' in data[0]:
        dataset_type = 'chartgalaxy'
        if args.image_folder is None:
            args.image_folder = "/mnt/VLAI_data/ChartGalaxy/images"
        
        # Normalize ChartGalaxy format: extract image filename from images list
        for item in data:
            if 'images' in item and isinstance(item['images'], list) and item['images']:
                # Remove subfolder prefix (e.g., "test_images/c174mz0m.png" -> "c174mz0m.png")
                img_path = item['images'][0]
                item['image'] = os.path.basename(img_path)
    else:
        dataset_type = 'infographicvqa'
    
    print(f"📊 Dataset: {dataset_type}")
    print(f"📁 Images: {args.image_folder}")

    output_filename = os.path.join(args.output_dir, f"{clean_model_name}_{dataset_type}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"📝 Running {len(data)} samples...")
    print(f"💾 Results will be saved to: {output_filename}")
    
    for item in tqdm(data, desc=f"{clean_model_name}"):
        img_path = os.path.join(args.image_folder, item['image'])
        
        if not os.path.exists(img_path):
            item["predict"] = "ERROR: Image not found"
            continue

        try:
            output = model.infer(item['question'], img_path)
            item["predict"] = output.split("ANSWER:")[-1].strip()
        except Exception as e:
            print(f"❌ Error on {item['image']}: {e}")
            item["predict"] = f"ERROR: {str(e)}"

    print(f"💾 Saving results to {output_filename}")
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("✅ Done!")
    return 0


if __name__ == "__main__":
    exit(main()) 