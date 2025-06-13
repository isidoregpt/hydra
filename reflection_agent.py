import asyncio

class ReflectionAgent:
    def __init__(self, anthropic_agent, max_parallel=1, delay_seconds=3):
        self.anthropic_agent = anthropic_agent
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.delay_seconds = delay_seconds

    async def critique(self, task, result):
        critique_prompt = f"""
You are an expert reviewer. Analyze the following task result for completeness, accuracy, and any missing considerations. If revisions are needed, suggest them.

Task: {task}
Result: {result}
"""
        async with self.semaphore:
            await asyncio.sleep(self.delay_seconds)  # Inject small delay to pace total token throughput
            critique = await self.anthropic_agent.chat(critique_prompt, max_tokens=512)
            return critique.strip()
