import anthropic

class AnthropicAgent:
    def __init__(self, api_key):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def chat(self, prompt, max_tokens=1024):
        response = await self.client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

