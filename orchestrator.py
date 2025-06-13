
import asyncio

class Orchestrator:
    def __init__(self, openai_agent, anthropic_agent, gemini_agent, planner_agent, debate_agent, console, subtask_limit, progress_callback=None):
        self.openai_agent = openai_agent
        self.anthropic_agent = anthropic_agent
        self.gemini_agent = gemini_agent
        self.planner_agent = planner_agent
        self.debate_agent = debate_agent
        self.console = console
        self.progress_callback = progress_callback
        self.subtask_limit = subtask_limit

    async def run(self, user_input):
        self.update_progress("Planning...")
        plan_steps = await self.planner_agent.generate_plan(user_input)
        plan_steps = plan_steps[:self.subtask_limit]
        full_results = {"Planning": plan_steps}

        for idx, step in enumerate(plan_steps, 1):
            self.update_progress(f"Running democracy on subtask {idx}/{len(plan_steps)}")

            # Phase 1: Generate independent responses
            task_prompt = f"Task: {step}"
            openai_response, anthropic_response, gemini_response = await asyncio.gather(
                self.openai_agent.chat(task_prompt),
                self.anthropic_agent.chat(task_prompt),
                self.gemini_agent.chat(task_prompt)
            )
            self.console.log(f"Subtask {idx} generated", model="OpenAI")
            self.console.log(f"Subtask {idx} generated", model="Claude")
            self.console.log(f"Subtask {idx} generated", model="Gemini")

            responses = {
                "OpenAI": openai_response,
                "Claude": anthropic_response,
                "Gemini": gemini_response
            }

            # Phase 2: Each model critiques peer responses
            critiques = {}
            critiques["OpenAI"] = await self.openai_agent.chat(self.critique_prompt(step, anthropic_response + "\n" + gemini_response))
            critiques["Claude"] = await self.anthropic_agent.chat(self.critique_prompt(step, openai_response + "\n" + gemini_response))
            critiques["Gemini"] = await self.gemini_agent.chat(self.critique_prompt(step, openai_response + "\n" + anthropic_response))

            # Phase 3: Arbitration
            final_answer = await self.debate_agent.arbitrate(step, responses, critiques)

            full_results[f"Step {idx}: {step}"] = {
                "Responses": responses,
                "Critiques": critiques,
                "Arbitrated Final": final_answer
            }

        self.update_progress("✅ Phase 5 Cognitive Democracy Complete")
        return full_results

    def critique_prompt(self, task, peer_outputs):
        return f"""
You are acting as a peer-review AI. Here are peer-generated outputs for the task. Provide constructive critique, point out flaws, suggest improvements.

Task: {task}

Peer Outputs:
{peer_outputs}
"""

    def update_progress(self, msg):
        if self.progress_callback:
            self.progress_callback(msg)
