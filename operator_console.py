class OperatorConsole:
    def __init__(self):
        self.logs = []
        self.usage = {"OpenAI": 0, "Claude": 0, "Gemini": 0}

    def log(self, msg, model=None):
        self.logs.append(msg)
        if model:
            self.usage[model] += 1

    def get_logs(self):
        return self.logs

    def get_usage(self):
        return self.usage
