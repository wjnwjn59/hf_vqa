"""LLaVA-OneVision Model Inference Module"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instances
_model: Optional[Any] = None
_processor: Optional[Any] = None

USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)


def get_model(model_dir: str = "/mnt/dataset1/pretrained_fm/lmms-lab_LLaVA-OneVision-1.5-4B-Instruct") -> Tuple[Any, Any]:
    """Initialize and return the LLaVA-OneVision model and processor."""
    global _model, _processor
    
    if _model is None or _processor is None:
        _processor = AutoProcessor.from_pretrained(
            model_dir, 
            trust_remote_code=True
        )
        
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True
        ).eval()
    
    return _model, _processor


def create_message_batch(questions: List[str], image_paths: List[str]) -> List[List[Dict[str, Any]]]:
    """
    Create batch messages for multiple samples.
    
    Format follows LLaVA-OneVision structure with nested content.
    """
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": f"{USER_PROMPT}\nQuestion: {question.strip()}\nAnswer:"}
                ]
            }
        ]
        for question, img_path in zip(questions, image_paths)
    ]


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """
    Run batch inference on LLaVA-OneVision model.
    
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
    
    # Apply chat template to the entire batch
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]
    
    # Process vision inputs using qwen_vl_utils
    image_inputs, video_inputs = process_vision_info(messages_batch)
    
    # Process batch inputs
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)
    
    # Generate for the entire batch
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=config.max_new_tokens
        )
    
    # Decode and remove input tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    decoded = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    # Get the first line of each response and strip whitespace
    return [text.splitlines()[0].strip() for text in decoded]

