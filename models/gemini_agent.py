import google.generativeai as genai
from typing import Optional

class GeminiAgent:
    def __init__(self, api_key, model_variant="2.5-flash"):
        genai.configure(api_key=api_key)
        self.model_variant = model_variant
        
        # Model mapping for different Gemini variants
        self.models = {
            "2.5-flash": 'gemini-2.5-flash',
            "2.5-pro": 'gemini-2.5-pro-preview-06-05', 
            "2.0-flash": 'gemini-2.0-flash-exp',
            "flash": 'gemini-2.5-flash',  # Alias
            "pro": 'gemini-2.5-pro-preview-06-05'  # Alias
        }
        
        # Default to 2.5 Flash
        model_name = self.models.get(model_variant, 'gemini-2.5-flash')
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
                    # Note: This is pseudocode - actual implementation depends on Google's API
                    # The thinking budget feature may require specific configuration
                    generation_config.thinking_budget = thinking_budgets[thinking_mode]
            
            response = await self.model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            return response.text.strip()
            
        except Exception as e:
            return f"Gemini API Error: {str(e)}"

    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            "variant": self.model_variant,
            "model_name": self.current_model_name,
            "supports_thinking": "2.5" in self.current_model_name,
            "is_pro": "pro" in self.model_variant,
            "is_flash": "flash" in self.model_variant
        }
