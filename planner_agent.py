class PlannerAgent:
    def __init__(self, gemini_agent):
        self.gemini_agent = gemini_agent

    async def generate_plan(self, user_input):
        planning_prompt = f"""
You are a cognitive planner. Break this complex problem into intelligent subtasks.

Task: {user_input}
"""
        plan = await self.gemini_agent.chat(planning_prompt)
        return [step for step in plan.strip().split('\n') if step.strip()]
