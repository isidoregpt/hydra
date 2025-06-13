class TaskManager:
    def __init__(self, gemini_agent):
        self.gemini_agent = gemini_agent

    def classify_task(self, user_input):
        prompt = f"Classify the following task into one of these categories: (Coding, Research, Writing, Summarization, Analysis, Creative Writing). Just output the category.\nTask: {user_input}"
        classification = self.gemini_agent.chat(prompt)
        return classification.strip()
