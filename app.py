
import streamlit as st
import uuid
import asyncio
from models.openai_agent import OpenAIAgent
from models.anthropic_agent import AnthropicAgent
from models.gemini_agent import GeminiAgent
from planner_agent import PlannerAgent
from reflection_agent import ReflectionAgent
from debate_agent import DebateAgent
from orchestrator import Orchestrator
from operator_console import OperatorConsole
from memory.memory_manager import MemoryManager

st.set_page_config(page_title="Hydra Phase 4.5 - Operator Console", layout="wide")
st.title("🧠 Hydra Phase 4.5 - Cognitive AI Orchestration")

with st.form("api_form"):
    openai_key = st.text_input("OpenAI API Key", type="password")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    gemini_key = st.text_input("Gemini API Key", type="password")
    submitted = st.form_submit_button("Save Keys")

if not all([openai_key, anthropic_key, gemini_key]):
    st.warning("Please enter all API keys to continue.")
    st.stop()

st.sidebar.header("⚙️ Operator Controls")
subtask_limit = st.sidebar.slider("Max Subtasks", min_value=1, max_value=100, value=15)

user_input = st.text_area("Enter your complex task for Hydra:")
memory = MemoryManager()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

status_placeholder = st.empty()
progress_bar = st.progress(0)
log_placeholder = st.empty()
usage_placeholder = st.empty()

if st.button("Run Hydra") and user_input:
    async def run_async():
        openai_agent = OpenAIAgent(openai_key)
        anthropic_agent = AnthropicAgent(anthropic_key)
        gemini_agent = GeminiAgent(gemini_key)
        planner_agent = PlannerAgent(gemini_agent)
        reflection_agent = ReflectionAgent(anthropic_agent)
        debate_agent = DebateAgent(gemini_agent, anthropic_agent)
        console = OperatorConsole()

        def progress_callback(msg):
            progress_bar.progress(min(1.0, (len(console.routing_log)+1)/(subtask_limit+2)))
            status_placeholder.info(msg)
            log_placeholder.write("\n".join(console.get_logs()))
            usage_placeholder.write(console.get_usage())

        orchestrator = Orchestrator(
            openai_agent, anthropic_agent, gemini_agent,
            planner_agent, reflection_agent, debate_agent,
            console, subtask_limit, progress_callback
        )

        result = await orchestrator.run(user_input)
        memory.save_session(st.session_state.session_id, result)

        st.header("🧠 Hydra Cognitive Output")
        for section, content in result.items():
            st.subheader(section)
            st.write(content)

    asyncio.run(run_async())
