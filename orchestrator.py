import asyncio

class Orchestrator:
    def __init__(self, openai_agent, anthropic_agent, gemini_agent, planner_agent, reflection_agent, debate_agent, progress_callback=None):
        self.openai_agent = openai_agent
        self.anthropic_agent = anthropic_agent
        self.gemini_agent = gemini_agent
        self.planner_agent = planner_agent
        self.reflection_agent = reflection_agent
        self.debate_agent = debate_agent
        self.progress_callback = progress_callback

    async def run(self, user_input):
        self.update_progress("Planning...")
        plan_steps = await self.planner_agent.generate_plan(user_input)
        full_results = {"Planning": plan_steps}

        for idx, step in enumerate(plan_steps, 1):
            self.update_progress(f"Executing subtask {idx}/{len(plan_steps)}: {step}")
            if "code" in step.lower():
                response = await self.openai_agent.chat(f"Solve: {step}")
            elif "analyze" in step.lower() or "research" in step.lower():
                response = await self.gemini_agent.chat(f"Analyze: {step}")
            else:
                response = await self.anthropic_agent.chat(f"Handle: {step}", max_tokens=512)

            self.update_progress(f"Reflecting on subtask {idx}/{len(plan_steps)}")
            critique_claude = await self.reflection_agent.critique(step, response)
            critique_gemini = await self.gemini_agent.chat(f"Critique this result: {response}")

            self.update_progress(f"Debating reflections on subtask {idx}/{len(plan_steps)}")
            merged_critique = await self.debate_agent.debate(step, critique_claude, critique_gemini)

            full_results[f"Step {idx}: {step}"] = {
                "Agent Response": response,
                "Critique Claude": critique_claude,
                "Critique Gemini": critique_gemini,
                "Merged Critique": merged_critique
            }

        self.update_progress("✅ Hydra Cognitive Cycle Complete")
        return full_results

    def update_progress(self, msg):
        if self.progress_callback:
            self.progress_callback(msg)
