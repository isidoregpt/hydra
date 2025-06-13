import streamlit as st
import uuid
import asyncio

from models.openai_agent import OpenAIAgent
from models.anthropic_agent import AnthropicAgent
from models.gemini_agent import GeminiAgent

from planner_agent import PlannerAgent
from debate_agent import DebateAgent
from orchestrator import Orchestrator
from operator_console import OperatorConsole
from memory.memory_manager import MemoryManager

st.set_page_config(page_title="Hydra Phase 5 - Model Democracy", layout="wide")
st.title("🧠 Hydra Phase 5 - Model Democracy & Consensus Arbitration")

with st.form("api_form"):
    openai_key = st.text_input("OpenAI API Key", type="password")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    gemini_key = st.text_input("Gemini API Key", type="password")
    submitted = st.form_submit_button("Save Keys")

if not all([openai_key, anthropic_key, gemini_key]):
    st.warning("Please enter all API keys to continue.")
    st.stop()

user_input = st.text_area("Enter your complex problem for Hydra:")
subtask_limit = st.slider("Subtask Limit", min_value=1, max_value=50, value=10)
memory = MemoryManager()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

status_placeholder = st.empty()
progress_bar = st.progress(0)

if st.button("Run Hydra") and user_input:
    async def run_async():
        openai_agent = OpenAIAgent(openai_key)
        anthropic_agent = AnthropicAgent(anthropic_key)
        gemini_agent = GeminiAgent(gemini_key)

        planner_agent = PlannerAgent(gemini_agent)
        debate_agent = DebateAgent(anthropic_agent)
        console = OperatorConsole()

        orchestrator = Orchestrator(
            openai_agent, anthropic_agent, gemini_agent,
            planner_agent, debate_agent, console,
            subtask_limit=subtask_limit,
            progress_callback=lambda msg: (status_placeholder.info(msg), progress_bar.progress(min(1.0, (console.usage['OpenAI'] + console.usage['Claude'] + console.usage['Gemini']) / (subtask_limit * 9))))
        )

        result = await orchestrator.run(user_input)
        memory.save_session(st.session_state.session_id, result)

        st.header("🧠 Final Hydra Output")
        for section, content in result.items():
            st.subheader(section)
            st.write(content)

        st.header("📊 Agent Usage Stats")
        st.write(console.get_usage())

        st.header("🪵 Full Cognitive Log")
        for log in console.get_logs():
            st.write(log)

    asyncio.run(run_async())
