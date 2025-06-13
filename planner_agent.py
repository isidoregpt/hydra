class PlannerAgent:
    def __init__(self, gemini_agent):
        self.gemini_agent = gemini_agent

    async def generate_plan(self, user_input):
        planning_prompt = f"""
You are a task planner. Break down the following task into sequential subtasks needed to solve it.
Format your response as numbered steps.

Task: {user_input}
"""
        plan = await self.gemini_agent.chat(planning_prompt)
        return plan.strip()
