import json
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from transformers import AutoProcessor


class Qwen3VLInference:
    """
    Qwen3-VL inference class using vLLM for vision-language tasks.
    """
    
    def __init__(
        self,
        model_name: str = 'Qwen/Qwen2-VL-7B-Instruct',
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize Qwen3-VL inference model.
        
        Args:
            model_name: Model name or path
            max_model_len: Maximum model length
            gpu_memory_utilization: GPU memory utilization ratio
            dtype: Data type for model
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            limit_mm_per_prompt: Limit multimodal inputs per prompt (e.g., {"image": 10})
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Default limit for multimodal inputs
        if limit_mm_per_prompt is None:
            limit_mm_per_prompt = {"image": 10}
        
        # Initialize vLLM model
        print(f"Loading model: {model_name}")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            limit_mm_per_prompt=limit_mm_per_prompt,
        )
        
        # Load processor
        print(f"Loading processor")
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    
    def format_messages(
        self, 
        text: str, 
        image_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Format messages with image for Qwen2-VL.
        
        Args:
            text: Text prompt
            image_path: Path to the image file
            
        Returns:
            Formatted messages list
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text", 
                        "text": text
                    },
                ],
            }
        ]
        
        return messages
    
    def generate(
        self, 
        prompts: List[str],
        image_paths: List[str],
        custom_sampling_params: Optional[SamplingParams] = None
    ) -> List[str]:
        """
        Generate responses for a batch of prompts with images.
        
        Args:
            prompts: List of text prompts
            image_paths: List of image file paths
            custom_sampling_params: Custom sampling parameters (optional)
            
        Returns:
            List of generated response strings
        """
        if len(prompts) != len(image_paths):
            raise ValueError("Number of prompts must match number of image paths")
        
        # Format messages for each prompt-image pair
        batch_messages = []
        for prompt, image_path in zip(prompts, image_paths):
            messages = self.format_messages(prompt, image_path)
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_messages.append({
                "prompt": text,
                "multi_modal_data": {
                    "image": image_path
                },
            })
        
        # Use custom sampling params if provided
        sampling_params = custom_sampling_params or self.sampling_params
        
        # Generate responses
        outputs = self.llm.generate(batch_messages, sampling_params)
        
        # Extract generated text
        responses = [output.outputs[0].text.strip() for output in outputs]
        
        return responses
    
    def generate_single(
        self, 
        prompt: str,
        image_path: str,
        custom_sampling_params: Optional[SamplingParams] = None
    ) -> str:
        """
        Generate response for a single prompt with image.
        
        Args:
            prompt: Text prompt
            image_path: Path to image file
            custom_sampling_params: Custom sampling parameters (optional)
            
        Returns:
            Generated response string
        """
        responses = self.generate(
            [prompt], 
            [image_path],
            custom_sampling_params=custom_sampling_params
        )
        return responses[0]
    
    def parse_json_response(self, generated_text: str) -> Dict[str, Any]:
        """
        Parse generated text to extract JSON response.
        
        Args:
            generated_text: Generated text from model
            
        Returns:
            Dictionary containing parsed information
        """
        try:
            content = generated_text.strip()
            
            # Try to extract JSON from the content
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                parsed_json = json.loads(json_str)
                return {
                    "response": parsed_json,
                    "success": True
                }
            else:
                return {
                    "response": content,
                    "success": False,
                    "error": "No valid JSON found in response"
                }
                
        except json.JSONDecodeError as e:
            return {
                "response": content if 'content' in locals() else generated_text,
                "success": False,
                "error": f"JSON decode error: {str(e)}"
            }
        except Exception as e:
            return {
                "response": generated_text,
                "success": False,
                "error": f"Parsing error: {str(e)}"
            }
    
    def update_sampling_params(
        self, 
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Update sampling parameters.
        
        Args:
            temperature: New temperature value
            top_p: New top_p value
            max_tokens: New max_tokens value
        """
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if max_tokens is not None:
            self.max_tokens = max_tokens
            
        # Update sampling params
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
