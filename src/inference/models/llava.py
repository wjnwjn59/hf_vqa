import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

from .base_model import VQAModel
from .utils import get_prompt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import transformers.modeling_flash_attention_utils as _mf

if not hasattr(_mf, "flash_attn_varlen_func"):
    def _flash_attn_varlen_func(*args, **kwargs):
        return _mf._flash_attention_forward(*args, **kwargs)
    _mf.flash_attn_varlen_func = _flash_attn_varlen_func

class LLaVAModel(VQAModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = "/mnt/dataset1/pretrained_fm/lmms-lab_LLaVA-OneVision-1.5-4B-Instruct"
        self._set_clean_model_name()
        self.load_model()

    def load_model(self):
        """Loads LLaVA-OneVision model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def infer(self, question: str, image_path: str):
        """Performs inference on image-question pair."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": get_prompt(question)}
            ]
        }]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(DEVICE)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text.strip()

