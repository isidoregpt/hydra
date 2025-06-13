class DebateAgent:
    def __init__(self, arbitration_agent):
        self.arbitration_agent = arbitration_agent

    async def arbitrate(self, task, responses, critiques):
        arbitration_prompt = f"""
You are an expert arbiter. Select and synthesize the best final answer based on these candidate responses and critiques.

Task: {task}

Responses:
- OpenAI: {responses['OpenAI']}
- Claude: {responses['Claude']}
- Gemini: {responses['Gemini']}

Critiques:
- OpenAI's critique: {critiques['OpenAI']}
- Claude's critique: {critiques['Claude']}
- Gemini's critique: {critiques['Gemini']}

Give your final best response that incorporates the strongest elements.
"""
        result = await self.arbitration_agent.chat(arbitration_prompt)
        return result.strip()
