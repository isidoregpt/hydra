import os
import json

class MemoryManager:
    def __init__(self, directory="memory/data_store"):
        os.makedirs(directory, exist_ok=True)
        self.directory = directory

    def save_session(self, session_id, data):
        file_path = os.path.join(self.directory, f"{session_id}.json")
        with open(file_path, "w") as f:
            json.dump(data, f)

    def load_session(self, session_id):
        file_path = os.path.join(self.directory, f"{session_id}.json")
        if not os.path.exists(file_path):
            return {}
        with open(file_path, "r") as f:
            return json.load(f)
