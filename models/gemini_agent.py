import google.generativeai as genai

class GeminiAgent:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    async def chat(self, prompt, temperature=0.7):
        """
        Async chat method for Gemini API calls.
        
        Args:
            prompt: The input prompt/question
            temperature: Controls randomness (0.0 to 1.0)
            
        Returns:
            String response from the model
        """
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature
            )
            
            response = await self.model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            return response.text.strip()
        except Exception as e:
            return f"Gemini API Error: {str(e)}"
