import anthropic

class AnthropicAgent:
    def __init__(self, api_key, model_variant="opus-4"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model_variant = model_variant
        
        # Claude 4 model mapping with official API names
        self.models = {
            "opus-4": "claude-opus-4-20250514",      # Most capable and intelligent
            "sonnet-4": "claude-sonnet-4-20250514",  # High-performance with exceptional reasoning
            "4": "claude-opus-4-20250514",           # Alias for Opus 4
            "claude-4": "claude-opus-4-20250514",    # Legacy alias
            # Backward compatibility
            "3.5-sonnet": "claude-3-5-sonnet-20241022",
            "sonnet": "claude-3-5-sonnet-20241022"
        }
        
        # Default to Claude Opus 4 (most capable)
        self.current_model = self.models.get(model_variant, "claude-opus-4-20250514")

    def switch_model(self, variant: str):
        """Switch to a different Claude model variant"""
        if variant in self.models:
            self.model_variant = variant
            self.current_model = self.models[variant]
            return True
        return False

    async def chat(self, prompt, max_tokens=4096, temperature=0.7):
        """
        Async chat method for Anthropic Claude 4 API calls.
        
        Args:
            prompt: The input prompt/question
            max_tokens: Maximum tokens in response (increased for Claude 4)
            temperature: Controls randomness (0.0 to 1.0)
            
        Returns:
            String response from the model
        """
        try:
            response = await self.client.messages.create(
                model=self.current_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Claude 4 API Error: {str(e)}"

    def get_model_info(self) -> dict:
        """Get information about the current Claude model"""
        model_info = {
            "claude-opus-4-20250514": {
                "name": "Claude Opus 4",
                "description": "Most powerful and capable model",
                "capabilities": ["Superior reasoning", "Complex analysis", "Advanced coding"],
                "context_window": 200000,
                "generation": 4,
                "tier": "flagship"
            },
            "claude-sonnet-4-20250514": {
                "name": "Claude Sonnet 4", 
                "description": "High-performance model with exceptional reasoning",
                "capabilities": ["Exceptional reasoning", "Efficient processing", "Balanced performance"],
                "context_window": 200000,
                "generation": 4,
                "tier": "performance"
            },
            "claude-3-5-sonnet-20241022": {
                "name": "Claude 3.5 Sonnet",
                "description": "Previous generation high-performance model",
                "capabilities": ["Strong reasoning", "Coding", "Analysis"],
                "context_window": 200000,
                "generation": 3.5,
                "tier": "legacy"
            }
        }
        
        return {
            "variant": self.model_variant,
            "model_name": self.current_model,
            "is_claude_4": "4-20250514" in self.current_model,
            "is_opus": "opus" in self.model_variant,
            "is_sonnet": "sonnet" in self.model_variant,
            **model_info.get(self.current_model, {
                "name": "Unknown Claude Model",
                "description": "Model information not available",
                "capabilities": [],
                "context_window": 200000,
                "generation": "unknown",
                "tier": "unknown"
            })
        }
