class Orchestrator:
    def __init__(self, openai_agent, anthropic_agent, gemini_agent, task_manager):
        self.openai_agent = openai_agent
        self.anthropic_agent = anthropic_agent
        self.gemini_agent = gemini_agent
        self.task_manager = task_manager

    def run(self, user_input):
        classification = self.task_manager.classify_task(user_input)

        # Dynamic orchestration based on classification
        if "Code" in classification:
            g_analysis = self.gemini_agent.chat(f"Analyze the following code task:\n{user_input}")
            o_thoughts = self.openai_agent.chat(f"Gemini says:\n{g_analysis}\nSuggest improvements or solutions.")
            a_summary = self.anthropic_agent.chat(f"Gemini says:\n{g_analysis}\nOpenAI says:\n{o_thoughts}\nSynthesize and finalize the best code plan.")
        else:
            g_analysis = self.gemini_agent.chat(f"Break down and analyze:\n{user_input}")
            o_thoughts = self.openai_agent.chat(f"Gemini says:\n{g_analysis}\nExpand, evaluate or critique as needed.")
            a_summary = self.anthropic_agent.chat(f"Gemini says:\n{g_analysis}\nOpenAI says:\n{o_thoughts}\nProvide the final expert answer.")

        return {
            "Task Type": classification,
            "Gemini Analysis": g_analysis,
            "OpenAI Thoughts": o_thoughts,
            "Claude Final Plan": a_summary
        }
