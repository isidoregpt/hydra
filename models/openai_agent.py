import openai

class OpenAIAgent:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def chat(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
