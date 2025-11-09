"""ChartGemma Model Inference Module"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instances
_model: Optional[PaliGemmaForConditionalGeneration] = None
_processor: Optional[AutoProcessor] = None

USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)


def get_model(model_dir: str = "/mnt/dataset1/pretrained_fm/ahmed-masry_chartgemma") -> Tuple[PaliGemmaForConditionalGeneration, AutoProcessor]:
    """
    Initialize and return the ChartGemma model and processor.
    
    ChartGemma is based on PaliGemma, fine-tuned for chart understanding tasks.
    """
    global _model, _processor
    
    if _model is None or _processor is None:
        # Load processor
        _processor = AutoProcessor.from_pretrained(model_dir)
        
        # Load model
        _model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16
        ).to(device)
        
        _model.eval()
    
    return _model, _processor


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """
    Run inference on ChartGemma model.
    
    ChartGemma is optimized for chart understanding and uses PaliGemma architecture.
    
    Args:
        questions: List of questions
        image_paths: List of paths to images (corresponding 1-to-1 with questions)
        config: DatasetConfig object containing max_new_tokens and other params
        
    Returns:
        List[str]: List of answers (stripped and first line taken)
    """
    model, processor = get_model()
    
    results = []
    
    # Process each question-image pair
    for question, image_path in zip(questions, image_paths):
        # Load and convert image to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Create input text with prompt and question
        input_text = f"{USER_PROMPT}\nQuestion: {question.strip()}\nAnswer:"
        
        # Process inputs
        inputs = processor(text=input_text, images=image, return_tensors="pt")
        prompt_length = inputs['input_ids'].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                num_beams=4,
                max_new_tokens=config.max_new_tokens
            )
        
        # Decode output (excluding prompt tokens)
        output_text = processor.batch_decode(
            generate_ids[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Get the first line and strip whitespace
        results.append(output_text.splitlines()[0].strip())
    
    return results

