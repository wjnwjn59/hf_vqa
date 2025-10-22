from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from .base_model import VQAModel
from .utils import get_prompt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    """Builds an image transformation pipeline."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: list, width: int, height: int, image_size: int):
    """Finds closest aspect ratio from list of targets."""
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
    """Preprocesses image by dividing into dynamic blocks."""
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


class InternVLModel(VQAModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = '/mnt/dataset1/pretrained_fm/OpenGVLab_InternVL3_5-8B'
        self._set_clean_model_name()
        self.image_size = 448
        self.transform = build_transform(self.image_size)
        self.load_model()

    def load_model(self):
        self.model = AutoModel.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False)

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
                self.tokenizer, pixel_values, prompt,
                generation_config={"max_new_tokens": 100, "pad_token_id": self.tokenizer.eos_token_id}
            )
        return response.strip() 