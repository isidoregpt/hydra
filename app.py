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
from memory.memory_manager import MemoryManager

st.set_page_config(page_title="Hydra Phase 4 - Cognitive AI", layout="wide")
st.title("🧠 Hydra Phase 4 - Autonomous Cognitive Reflection")

with st.form("api_form"):
    openai_key = st.text_input("OpenAI API Key", type="password")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    gemini_key = st.text_input("Gemini API Key", type="password")
    submitted = st.form_submit_button("Save Keys")

if not all([openai_key, anthropic_key, gemini_key]):
    st.warning("Please enter all API keys to continue.")
    st.stop()

user_input = st.text_area("Enter your complex problem for Hydra:")
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
        reflection_agent = ReflectionAgent(anthropic_agent)
        debate_agent = DebateAgent(gemini_agent, anthropic_agent)

        # Progress feedback
        step_counter = {"step": 0}
        total_phases = 5

        def progress_callback(msg):
            step_counter["step"] += 1
            pct = step_counter["step"] / (len(user_input.split()) + total_phases)
            progress_bar.progress(min(pct, 1.0))
            status_placeholder.info(msg)

        orchestrator = Orchestrator(
            openai_agent, anthropic_agent, gemini_agent,
            planner_agent, reflection_agent, debate_agent, progress_callback
        )
        result = await orchestrator.run(user_input)
        memory.save_session(st.session_state.session_id, result)

        st.header("🧠 Hydra Cognitive Output")
        for section, content in result.items():
            st.subheader(section)
            st.write(content)

    asyncio.run(run_async())
