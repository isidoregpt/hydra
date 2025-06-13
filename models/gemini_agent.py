import google.generativeai as genai

class GeminiAgent:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro-preview-06-05')

    async def chat(self, prompt):
        response = await self.model.generate_content_async(prompt)
        return response.text.strip()
