from pathlib import Path
import os
import glob
from typing import Dict

from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import safetensors.torch as st

from .base_model import VQAModel
from .utils import get_prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list, width: int, height: int, image_size: int):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_ar)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )
    target_ar = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_ar[0]
    target_height = image_size * target_ar[1]
    blocks = target_ar[0] * target_ar[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _load_safetensor_state_dict(folder: str) -> Dict[str, torch.Tensor]:
    """Load state_dict from safetensors shards."""
    single = os.path.join(folder, "model.safetensors")
    if os.path.exists(single):
        print(f"📦 Loading single safetensor: {single}")
        return st.load_file(single)
    
    shards = sorted(glob.glob(os.path.join(folder, "model-*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No model*.safetensors found in: {folder}")
    
    print(f"📦 Loading {len(shards)} safetensor shards...")
    state: Dict[str, torch.Tensor] = {}
    for f in shards:
        state.update(st.load_file(f))
    return state


class InternVLModel(VQAModel):
    """
    InternVL model with smart fallback:
    1. Try loading merged model directly
    2. If no chat() method, load architecture from base model + weights from merged
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Merged model path (from LLaMA Factory export)
        self.merged_path = kwargs.get(
            "model_path",
            "/home/khoina/BizGen/hf_vqa/src/training/LLaMA-Factory/output/intern3_5vl_lora_sft_621",
        )

        # Base model path (for architecture/code)
        self.base_model_path = kwargs.get(
            "base_model_path",
            "/mnt/dataset1/pretrained_fm/OpenGVLab_InternVL3_5-2B",  # Fixed: use 8B not 2B
        )

        # Tokenizer path (prefer merged for added_tokens/chat_template)
        self.tokenizer_path = kwargs.get("tokenizer_path", self.merged_path)

        self.model_path = self.merged_path  # For _set_clean_model_name
        self._set_clean_model_name()
        self.image_size = 448
        self.transform = build_transform(self.image_size)
        self.load_model()

    def load_model(self):
        print(f"🔍 Attempting to load merged model from: {self.merged_path}")
        
        # Try loading merged model directly
        try:
            self.model = AutoModel.from_pretrained(
                self.merged_path,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
            ).eval()
            
            print(f"✅ Model loaded: {type(self.model).__name__}")
            
            # Check if chat method exists
            if hasattr(self.model, "chat"):
                print("✅ Model has chat() method - ready to use")
                use_fallback = False
            else:
                print("⚠️  Model missing chat() method - will use fallback")
                use_fallback = True
                
        except Exception as e:
            print(f"⚠️  Failed to load merged model: {e}")
            print("🔄 Will use fallback method")
            use_fallback = True
            self.model = None
        
        # Fallback: Load architecture from base model + weights from merged
        if use_fallback:
            print(f"🔄 Loading architecture from base model: {self.base_model_path}")
            
            # Load base model with architecture/code
            model_with_code = AutoModel.from_pretrained(
                self.base_model_path,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,  # Ensure consistent dtype
            )
            
            # Verify it has chat method
            if not hasattr(model_with_code, "chat"):
                raise AttributeError(f"Base model at {self.base_model_path} also missing chat() method!")
            
            print(f"📦 Loading merged weights from: {self.merged_path}")
            # Load merged weights
            sd = _load_safetensor_state_dict(self.merged_path)
            
            # Load state dict with warnings
            missing, unexpected = model_with_code.load_state_dict(sd, strict=False)
            
            if missing:
                print(f"⚠️  Missing keys: {len(missing)}")
                if len(missing) <= 10:
                    for k in missing:
                        print(f"   - {k}")
            
            if unexpected:
                print(f"⚠️  Unexpected keys: {len(unexpected)}")
                if len(unexpected) <= 10:
                    for k in unexpected:
                        print(f"   - {k}")
            
            # Move to device and eval mode
            self.model = model_with_code.to(device).eval()
            print("✅ Fallback successful - model ready")

        # Load tokenizer
        print(f"🔤 Loading tokenizer from: {self.tokenizer_path}")
        tok_path = Path(self.tokenizer_path)
        
        # Check if tokenizer files exist
        tokenizer_exists = any([
            (tok_path / "tokenizer.json").exists(),
            (tok_path / "vocab.json").exists(),
            (tok_path / "tokenizer.model").exists(),
            (tok_path / "tokenizer_config.json").exists(),
        ])
        
        tokenizer_source = str(tok_path) if tokenizer_exists else self.base_model_path
        
        if tokenizer_source != str(tok_path):
            print(f"⚠️  Tokenizer not found at {tok_path}, using base model tokenizer")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
            use_fast=False,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Tokenizer loaded from: {tokenizer_source}")

    def _load_image(self, image_file: str):
        """Loads and preprocesses image."""
        image = Image.open(image_file).convert('RGB')
        images = dynamic_preprocess(image, image_size=self.image_size, use_thumbnail=True, max_num=12)
        pixel_values = [self.transform(img) for img in images]
        return torch.stack(pixel_values)

    def infer(self, question: str, image_path: str):
        """Performs inference on image-question pair."""
        pixel_values = self._load_image(image_path).to(torch.bfloat16).to(device)
        prompt = f"<image>{get_prompt(question)}"

        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config={"max_new_tokens": 100, "pad_token_id": self.tokenizer.eos_token_id},
            )
        return response.strip()