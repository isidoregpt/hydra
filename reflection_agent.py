class ReflectionAgent:
    def __init__(self, anthropic_agent):
        self.anthropic_agent = anthropic_agent

    async def critique(self, task, result):
        critique_prompt = f"""
You are an expert cognitive agent. Review the following output for completeness, correctness, logic gaps, or any flaws.

Task: {task}
Result: {result}
"""
        critique = await self.anthropic_agent.chat(critique_prompt, max_tokens=512)
        return critique.strip()
