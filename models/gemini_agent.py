import google.generativeai as genai

class GeminiAgent:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')

    def chat(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text
