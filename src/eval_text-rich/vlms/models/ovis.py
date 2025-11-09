"""Ovis1.5-Gemma2 Model Inference Module"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instances
_model: Optional[Any] = None
_text_tokenizer: Optional[Any] = None
_visual_tokenizer: Optional[Any] = None
_conversation_formatter: Optional[Any] = None

USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)


def get_model(model_dir: str = "/mnt/dataset1/pretrained_fm/AIDC-AI_Ovis1.5-Gemma2-9B") -> Tuple[Any, Any, Any, Any]:
    """
    Initialize and return the Ovis1.5-Gemma2 model and its components.
    
    Ovis uses a unique architecture with separate text and visual tokenizers.
    """
    global _model, _text_tokenizer, _visual_tokenizer, _conversation_formatter
    
    if _model is None:
        # Load model
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            multimodal_max_length=8192,
            trust_remote_code=True
        ).cuda()
        
        # Get tokenizers and formatter from model
        _text_tokenizer = _model.get_text_tokenizer()
        _visual_tokenizer = _model.get_visual_tokenizer()
        _conversation_formatter = _model.get_conversation_formatter()
    
    return _model, _text_tokenizer, _visual_tokenizer, _conversation_formatter


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """
    Run inference on Ovis1.5-Gemma2 model.
    
    Args:
        questions: List of questions
        image_paths: List of paths to images (corresponding 1-to-1 with questions)
        config: DatasetConfig object containing max_new_tokens and other params
        
    Returns:
        List[str]: List of answers (stripped and first line taken)
    """
    model, text_tokenizer, visual_tokenizer, conversation_formatter = get_model()
    
    results = []
    
    # Process each question-image pair
    for question, image_path in zip(questions, image_paths):
        # Load image
        image = Image.open(image_path)
        
        # Create query with image tag and prompt
        query = f'<image>\n{USER_PROMPT}\nQuestion: {question.strip()}\nAnswer:'
        
        # Format query using conversation formatter
        prompt, input_ids = conversation_formatter.format_query(query)
        input_ids = torch.unsqueeze(input_ids, dim=0).to(device=model.device)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).to(device=model.device)
        
        # Preprocess image with visual tokenizer
        pixel_values = [visual_tokenizer.preprocess_image(image).to(
            dtype=visual_tokenizer.dtype, 
            device=visual_tokenizer.device
        )]
        
        # Generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True
            )
            
            output_ids = model.generate(
                input_ids, 
                pixel_values=pixel_values, 
                attention_mask=attention_mask, 
                **gen_kwargs
            )[0]
            
            # Decode output
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Get the first line and strip whitespace
        results.append(output.splitlines()[0].strip())
    
    return results
