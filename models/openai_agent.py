import openai

class OpenAIAgent:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    async def chat(self, prompt):
        response = await self.client.chat.completions.acreate(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
