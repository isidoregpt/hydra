class ReflectionAgent:
    def __init__(self, agent):
        self.agent = agent

    async def critique(self, task, peer_response):
        critique_prompt = f"""
You are acting as a peer-review AI. Evaluate this response for logic, accuracy, and completeness.

Task: {task}
Response: {peer_response}
"""
        critique = await self.agent.chat(critique_prompt)
        return critique.strip()
