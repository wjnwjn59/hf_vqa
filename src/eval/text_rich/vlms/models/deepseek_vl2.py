"""DeepSeek-VL2 Model Inference Module"""

import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instances
_model: Optional[DeepseekVLV2ForCausalLM] = None
_vl_chat_processor: Optional[DeepseekVLV2Processor] = None
_tokenizer: Optional[Any] = None

USER_PROMPT = (
    "Answer each question concisely in a single word or short phrase, without any lengthy descriptions or explanations.\n"
    "Rely only on information that is clearly visible in the provided image.\n"
    "If the answer cannot be determined from the image, respond with 'unanswerable'."
)


def get_model(model_dir: str = "/mnt/dataset1/pretrained_fm/deepseek-ai_deepseek-vl2-small") -> Tuple[DeepseekVLV2ForCausalLM, DeepseekVLV2Processor, Any]:
    """Initialize and return the DeepSeek-VL2 model, processor, and tokenizer."""
    global _model, _vl_chat_processor, _tokenizer
    
    if _model is None or _vl_chat_processor is None or _tokenizer is None:
        # Load VL chat processor
        _vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_dir)
        _tokenizer = _vl_chat_processor.tokenizer
        
        # Load model
        _model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        _model = _model.to(torch.bfloat16).cuda().eval()
    
    return _model, _vl_chat_processor, _tokenizer


def create_conversation(question: str, image_path: str) -> List[Dict[str, Any]]:
    """
    Create conversation format for DeepSeek-VL2.
    
    Format follows DeepSeek-VL2 structure with special role tags.
    Note: Uses <image> tag instead of <image_placeholder> for single image.
    """
    return [
        {
            "role": "<|User|>",
            "content": f"<image>\n{USER_PROMPT}\nQuestion: {question.strip()}\nAnswer:",
            "images": [image_path],
        },
        {
            "role": "<|Assistant|>",
            "content": "",
        },
    ]


def inference(questions: List[str], image_paths: List[str], config: Dict[str, Any]) -> List[str]:
    """
    Run inference on DeepSeek-VL2 model.
    
    Note: DeepSeek-VL2 processes samples one at a time based on the provided example code.
    The force_batchify=True in the processor is for internal batching, not multi-sample batching.
    
    Args:
        questions: List of questions
        image_paths: List of paths to images (corresponding 1-to-1 with questions)
        config: DatasetConfig object containing max_new_tokens and other params
        
    Returns:
        List[str]: List of answers (stripped and first line taken)
    """
    model, vl_chat_processor, tokenizer = get_model()
    
    results = []
    
    # Process each question-image pair
    for question, image_path in zip(questions, image_paths):
        # Create conversation
        conversation = create_conversation(question, image_path)
        
        # Load images and prepare inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(model.device)
        
        # Run image encoder to get the image embeddings
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        
        # Generate response
        with torch.no_grad():
            outputs = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        
        # Decode response
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        # Get the first line and strip whitespace
        results.append(answer.splitlines()[0].strip())
    
    return results

