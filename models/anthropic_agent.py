import anthropic

class AnthropicAgent:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)

    def chat(self, prompt):
        response = self.client.messages.create(
            model="claude-3.5-sonnet-20240610",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
