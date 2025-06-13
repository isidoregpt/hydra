import asyncio

class Orchestrator:
    def __init__(self, openai_agent, anthropic_agent, gemini_agent, planner_agent, reflection_agent):
        self.openai_agent = openai_agent
        self.anthropic_agent = anthropic_agent
        self.gemini_agent = gemini_agent
        self.planner_agent = planner_agent
        self.reflection_agent = reflection_agent

    async def run(self, user_input):
        plan = await self.planner_agent.generate_plan(user_input)
        plan_steps = plan.split('\n')
        full_results = {}
        full_results['Planning Breakdown'] = plan

        async def handle_step(idx, step):
            if "code" in step.lower():
                response = await self.openai_agent.chat(f"Solve this code task step:\n{step}")
            elif "analyze" in step.lower() or "research" in step.lower():
                response = await self.gemini_agent.chat(f"Analyze this step:\n{step}")
            else:
                response = await self.anthropic_agent.chat(f"Handle this task step:\n{step}", max_tokens=512)

            critique = await self.reflection_agent.critique(step, response)
            return idx, step, response, critique

        tasks = [
            handle_step(idx, step.strip())
            for idx, step in enumerate(plan_steps, start=1) if step.strip()
        ]

        results = await asyncio.gather(*tasks)

        for idx, step, response, critique in results:
            full_results[f"Step {idx}: {step}"] = {
                "Agent Response": response,
                "Critique": critique
            }

        return full_results
