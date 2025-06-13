class Orchestrator:
    def __init__(self, openai_agent, anthropic_agent, gemini_agent, planner_agent, reflection_agent):
        self.openai_agent = openai_agent
        self.anthropic_agent = anthropic_agent
        self.gemini_agent = gemini_agent
        self.planner_agent = planner_agent
        self.reflection_agent = reflection_agent

    def run(self, user_input):
        plan = self.planner_agent.generate_plan(user_input)
        plan_steps = plan.split('\n')
        full_results = {}
        full_results['Planning Breakdown'] = plan

        for idx, step in enumerate(plan_steps, start=1):
            step = step.strip()
            if not step:
                continue

            # Assign agents based on simple rule (this is your first router logic)
            if "code" in step.lower():
                response = self.openai_agent.chat(f"Solve this code task step:\n{step}")
            elif "analyze" in step.lower() or "research" in step.lower():
                response = self.gemini_agent.chat(f"Analyze this step:\n{step}")
            else:
                response = self.anthropic_agent.chat(f"Handle this task step:\n{step}")

            critique = self.reflection_agent.critique(step, response)

            full_results[f"Step {idx}: {step}"] = {
                "Agent Response": response,
                "Critique": critique
            }

        return full_results
