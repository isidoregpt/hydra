import google.generativeai as genai
from typing import Optional

class GeminiAgent:
    def __init__(self, api_key, model_variant="2.5-flash"):
        genai.configure(api_key=api_key)
        self.model_variant = model_variant
        
        # Updated model mapping with OFFICIAL Gemini 2.5 model names from Google AI API
        self.models = {
            "2.5-flash": 'gemini-2.5-flash-preview-05-20',
            "2.5-pro": 'gemini-2.5-pro-preview-06-05',
            "2.0-flash": 'gemini-2.0-flash',
            "2.0-flash-lite": 'gemini-2.0-flash-lite',
            "flash": 'gemini-2.5-flash-preview-05-20',  # Default alias
            "pro": 'gemini-2.5-pro-preview-06-05',  # Default alias
            # Backward compatibility aliases
            "1.5-flash": 'gemini-2.5-flash-preview-05-20',  # Upgrade to 2.5
            "1.5-pro": 'gemini-2.5-pro-preview-06-05'  # Upgrade to 2.5
        }
        
        # Default to 2.5 Flash (best price-performance)
        model_name = self.models.get(model_variant, 'gemini-2.5-flash-preview-05-20')
        self.model = genai.GenerativeModel(model_name)
        self.current_model_name = model_name

    def switch_model(self, variant: str):
        """Switch to a different Gemini model variant"""
        if variant in self.models:
            self.model_variant = variant
            model_name = self.models[variant]
            self.model = genai.GenerativeModel(model_name)
            self.current_model_name = model_name
            return True
        return False

    async def chat(self, prompt: str, temperature: float = 0.7, thinking_mode: Optional[str] = None):
        """
        Async chat method for Gemini API calls with thinking mode support.
        
        Args:
            prompt: The input prompt/question
            temperature: Controls randomness (0.0 to 1.0)
            thinking_mode: Thinking depth for 2.5 models (minimal, low, medium, high, max)
            
        Returns:
            String response from the model
        """
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature
            )
            
            # Add thinking mode for 2.5 models that support it
            if thinking_mode and "2.5" in self.current_model_name:
                thinking_budgets = {
                    "minimal": 128,
                    "low": 2048, 
                    "medium": 8192,
                    "high": 16384,
                    "max": 32768
                }
                
                if thinking_mode in thinking_budgets:
                    # Configure thinking budget for 2.5 models
                    # Note: The exact parameter name may vary - check Google's latest API docs
                    generation_config.thinking_budget = thinking_budgets[thinking_mode]
            
            response = await self.model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            return response.text.strip()
            
        except Exception as e:
            # If thinking budget parameter is not recognized, try without it
            if "thinking_budget" in str(e) and thinking_mode:
                try:
                    generation_config = genai.types.GenerationConfig(temperature=temperature)
                    response = await self.model.generate_content_async(
                        prompt,
                        generation_config=generation_config
                    )
                    return response.text.strip()
                except Exception as e2:
                    return f"Gemini API Error: {str(e2)}"
            return f"Gemini API Error: {str(e)}"

    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            "variant": self.model_variant,
            "model_name": self.current_model_name,
            "supports_thinking": "2.5" in self.current_model_name,
            "is_pro": "pro" in self.model_variant,
            "is_flash": "flash" in self.model_variant,
            "is_2_5": "2.5" in self.current_model_name,
            "is_2_0": "2.0" in self.current_model_name
        }
