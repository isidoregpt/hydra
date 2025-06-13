class OperatorConsole:
    def __init__(self):
        self.routing_log = []
        self.agent_usage = {"OpenAI": 0, "Gemini": 0, "Claude": 0}

    def log(self, subtask, model, phase):
        self.routing_log.append(f"{phase} on '{subtask}' via {model}")
        self.agent_usage[model] += 1

    def get_logs(self):
        return self.routing_log

    def get_usage(self):
        return self.agent_usage
