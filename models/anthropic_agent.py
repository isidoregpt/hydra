import anthropic

class AnthropicAgent:
    def __init__(self, api_key):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def chat(self, prompt, max_tokens=2048, temperature=0.7):
        """
        Async chat method for Anthropic API calls.
        
        Args:
            prompt: The input prompt/question
            max_tokens: Maximum tokens in response
            temperature: Controls randomness (0.0 to 1.0)
            
        Returns:
            String response from the model
        """
        try:
            response = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"Anthropic API Error: {str(e)}"
