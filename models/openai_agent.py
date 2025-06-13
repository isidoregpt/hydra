import openai
import asyncio

class OpenAIAgent:
    def __init__(self, api_key):
        # Use AsyncOpenAI for proper async support
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def chat(self, prompt, max_tokens=1024, temperature=0.7):
        """
        Async chat method for OpenAI API calls.
        
        Args:
            prompt: The input prompt/question
            max_tokens: Maximum tokens in response
            temperature: Controls randomness (0.0 to 1.0)
            
        Returns:
            String response from the model
        """
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"OpenAI API Error: {str(e)}"
