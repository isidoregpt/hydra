class Orchestrator:
    def __init__(self, openai_agent, anthropic_agent, gemini_agent):
        self.openai_agent = openai_agent
        self.anthropic_agent = anthropic_agent
        self.gemini_agent = gemini_agent

    def run(self, user_input):
        g_analysis = self.gemini_agent.chat(f"Analyze and break down the problem:\n{user_input}")
        o_thoughts = self.openai_agent.chat(f"Gemini analysis:\n{g_analysis}\nWhat do you recommend?")
        a_summary = self.anthropic_agent.chat(f"Gemini says:\n{g_analysis}\nOpenAI says:\n{o_thoughts}\nGive a final expert plan.")
        return {
            "Gemini Analysis": g_analysis,
            "OpenAI Thoughts": o_thoughts,
            "Claude Final Plan": a_summary
        }
