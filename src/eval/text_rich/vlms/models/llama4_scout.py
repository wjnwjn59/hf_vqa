"""Llama-4-Scout Model Inference Module"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instances
_model: Optional[Llama4ForConditionalGeneration] = None
_processor: Optional[AutoProcessor] = None

USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)


def get_model(model_dir: str = "/mnt/dataset1/pretrained_fm/unsloth_Llama-4-Scout-17B-16E-Instruct") -> Tuple[Llama4ForConditionalGeneration, AutoProcessor]:
    """Initialize and return the Llama-4-Scout model and processor."""
    global _model, _processor
    
    if _model is None or _processor is None:
        _processor = AutoProcessor.from_pretrained(model_dir)
        
        _model = Llama4ForConditionalGeneration.from_pretrained(
            model_dir,
            attn_implementation="flex_attention",
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
    
    return _model, _processor


def create_message_batch(questions: List[str], image_paths: List[str]) -> List[List[Dict[str, Any]]]:
    """
    Create batch messages for multiple samples.
    
    Format follows Llama-4 Vision structure with image URL and text content.
    """
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": f"file://{img_path}"},
                    {"type": "text", "text": f"{USER_PROMPT}\nQuestion: {question.strip()}\nAnswer:"}
                ]
            }
        ]
        for question, img_path in zip(questions, image_paths)
    ]


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """
    Run batch inference on Llama-4-Scout model.
    
    Args:
        questions: List of questions
        image_paths: List of paths to images (corresponding 1-to-1 with questions)
        config: DatasetConfig object containing max_new_tokens and other params
        
    Returns:
        List[str]: List of answers (stripped and first line taken)
    """
    model, processor = get_model()
    
    # Create batch messages
    messages_batch = create_message_batch(questions, image_paths)
    
    # Process batch with apply_chat_template directly
    results = []
    for messages in messages_batch:
        # Apply chat template with tokenization
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
            )
        
        # Decode response (excluding input tokens)
        response = processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Get the first line and strip whitespace
        results.append(response.splitlines()[0].strip())
    
    return results

