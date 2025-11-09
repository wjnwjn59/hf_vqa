import torch
from transformers import AutoModelForCausalLM
from PIL import Image

from .base_model import VQAModel
from .utils import get_prompt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OvisModel(VQAModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = "/mnt/dataset1/pretrained_fm/AIDC-AI_Ovis2.5-9B"
        self._set_clean_model_name()
        self.load_model()

    def load_model(self):
        """Loads Ovis model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16,
            multimodal_max_length=32768, trust_remote_code=True
        ).eval().to(DEVICE)

    def infer(self, question: str, image_path: str):
        """Performs inference on image-question pair."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path).convert('RGB')},
                {"type": "text", "text": get_prompt(question)},
            ],
        }]
        
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = input_ids.cuda()
        pixel_values = pixel_values.cuda() if pixel_values is not None else None
        grid_thws = grid_thws.cuda() if grid_thws is not None else None
        
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs=input_ids, pixel_values=pixel_values, grid_thws=grid_thws,
                enable_thinking=False, enable_thinking_budget=False,
                max_new_tokens=100, thinking_budget=0,
                eos_token_id=self.model.text_tokenizer.eos_token_id
            )
        
        response = self.model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip() 