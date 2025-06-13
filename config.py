import os

class Config:
    @staticmethod
    def get_api_keys():
        return {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        }
