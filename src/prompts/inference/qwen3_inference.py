import json
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class Qwen3Inference:
    """
    Qwen 3 inference class using vLLM without thinking mode.
    """
    
    def __init__(
        self,
        model_name: str = 'unsloth/Qwen3-8B',
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
    ):
        """
        Initialize Qwen3 inference model.
        
        Args:
            model_name: Model name or path
            max_model_len: Maximum model length
            gpu_memory_utilization: GPU memory utilization ratio
            dtype: Data type for model
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Initialize vLLM model
        print(f"Loading model: {model_name}")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype
        )
        
        # Load tokenizer
        print(f"Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[151645, 151668]  # Stop tokens for Qwen3
        )
    
    def format_prompt(self, content: str, enable_thinking: bool = False) -> str:
        """
        Format prompt using chat template.
        
        Args:
            content: User message content
            enable_thinking: Whether to enable thinking mode (default: False)
            
        Returns:
            Formatted prompt string
        """
        messages = [{"role": "user", "content": content}]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        return formatted_prompt
    
    def generate(
        self, 
        prompts: List[str], 
        enable_thinking: bool = False,
        custom_sampling_params: Optional[SamplingParams] = None
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            enable_thinking: Whether to enable thinking mode
            custom_sampling_params: Custom sampling parameters (optional)
            
        Returns:
            List of generated response strings
        """
        # Format prompts
        formatted_prompts = [
            self.format_prompt(prompt, enable_thinking) 
            for prompt in prompts
        ]
        
        # Use custom sampling params if provided
        sampling_params = custom_sampling_params or self.sampling_params
        
        # Generate responses
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
        # Extract generated text
        responses = [output.outputs[0].text.strip() for output in outputs]
        
        return responses
    
    def generate_single(
        self, 
        prompt: str, 
        enable_thinking: bool = False,
        custom_sampling_params: Optional[SamplingParams] = None
    ) -> str:
        """
        Generate response for a single prompt.
        
        Args:
            prompt: Prompt string
            enable_thinking: Whether to enable thinking mode
            custom_sampling_params: Custom sampling parameters (optional)
            
        Returns:
            Generated response string
        """
        responses = self.generate(
            [prompt], 
            enable_thinking=enable_thinking,
            custom_sampling_params=custom_sampling_params
        )
        return responses[0]
    
    def parse_json_response(self, generated_text: str) -> Dict[str, Any]:
        """
        Parse generated text to extract thinking content and JSON response.
        
        Args:
            generated_text: Generated text from model
            
        Returns:
            Dictionary containing parsed information
        """
        try:
            # Find the </think> token position if present
            if "</think>" in generated_text:
                think_end = generated_text.find("</think>")
                thinking_content = generated_text[:think_end].replace("<think>", "").strip()
                content = generated_text[think_end + len("</think>"):].strip()
            else:
                thinking_content = ""
                content = generated_text
            
            # Try to extract JSON from the content
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                parsed_json = json.loads(json_str)
                return {
                    "thinking": thinking_content,
                    "response": parsed_json,
                    "success": True
                }
            else:
                return {
                    "thinking": thinking_content,
                    "response": content,
                    "success": False,
                    "error": "No valid JSON found in response"
                }
                
        except json.JSONDecodeError as e:
            return {
                "thinking": thinking_content if 'thinking_content' in locals() else "",
                "response": content if 'content' in locals() else generated_text,
                "success": False,
                "error": f"JSON decode error: {str(e)}"
            }
        except Exception as e:
            return {
                "thinking": "",
                "response": generated_text,
                "success": False,
                "error": f"Parsing error: {str(e)}"
            }
    
    def generate_and_parse_json(
        self, 
        prompts: List[str], 
        enable_thinking: bool = False,
        custom_sampling_params: Optional[SamplingParams] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate responses and parse JSON for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            enable_thinking: Whether to enable thinking mode
            custom_sampling_params: Custom sampling parameters (optional)
            
        Returns:
            List of parsed response dictionaries
        """
        responses = self.generate(
            prompts, 
            enable_thinking=enable_thinking,
            custom_sampling_params=custom_sampling_params
        )
        
        parsed_responses = [
            self.parse_json_response(response) 
            for response in responses
        ]
        
        return parsed_responses
    
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
            stop_token_ids=[151645, 151668]
        )