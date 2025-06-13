import streamlit as st
import uuid
from models.openai_agent import OpenAIAgent
from models.anthropic_agent import AnthropicAgent
from models.gemini_agent import GeminiAgent
from task_manager import TaskManager
from orchestrator import Orchestrator
from memory.memory_manager import MemoryManager

st.title("🧠 Hydra Phase 2 - Dynamic Multi-Model AI Orchestrator")

with st.form("api_form"):
    openai_key = st.text_input("OpenAI API Key", type="password")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    gemini_key = st.text_input("Gemini API Key", type="password")
    submitted = st.form_submit_button("Save Keys")

if not all([openai_key, anthropic_key, gemini_key]):
    st.warning("Please enter all API keys to continue.")
    st.stop()

user_input = st.text_area("Enter your problem/task for Hydra to solve:")
memory = MemoryManager()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if st.button("Run Hydra") and user_input:
    openai_agent = OpenAIAgent(openai_key)
    anthropic_agent = AnthropicAgent(anthropic_key)
    gemini_agent = GeminiAgent(gemini_key)
    task_manager = TaskManager(gemini_agent)
    orchestrator = Orchestrator(openai_agent, anthropic_agent, gemini_agent, task_manager)

    result = orchestrator.run(user_input)
    memory.save_session(st.session_state.session_id, result)

    st.header("🧠 Multi-Agent Response")
    for stage, output in result.items():
        st.subheader(stage)
        st.write(output)
