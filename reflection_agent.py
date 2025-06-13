class ReflectionAgent:
    def __init__(self, anthropic_agent, semaphore):
        self.anthropic_agent = anthropic_agent
        self.semaphore = semaphore

    async def critique(self, task, result):
        critique_prompt = f"""
You are an expert reviewer. Analyze the following task result for completeness, accuracy, and any missing considerations. If revisions are needed, suggest them.

Task: {task}
Result: {result}
"""
        async with self.semaphore:
            critique = await self.anthropic_agent.chat(critique_prompt)
            return critique.strip()
