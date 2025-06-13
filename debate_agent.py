class DebateAgent:
    def __init__(self, gemini_agent, anthropic_agent):
        self.gemini_agent = gemini_agent
        self.anthropic_agent = anthropic_agent

    async def debate(self, task, critique_a, critique_b):
        debate_prompt = f"""
Compare and synthesize these two critiques about the same task result. Merge their insights, resolve conflicts, and suggest a combined improvement path.

Task: {task}

Critique 1 (Claude): {critique_a}
Critique 2 (Gemini): {critique_b}
"""
        merged = await self.anthropic_agent.chat(debate_prompt, max_tokens=512)
        return merged.strip()

