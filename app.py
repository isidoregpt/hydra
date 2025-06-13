import streamlit as st
from models.openai_agent import OpenAIAgent
from models.anthropic_agent import AnthropicAgent
from models.gemini_agent import GeminiAgent
from orchestrator import Orchestrator

st.title("🧠 Hydra v2 - Real-time Multi-Model AI Collaboration")

with st.form("api_form"):
    openai_key = st.text_input("OpenAI API Key", type="password")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    gemini_key = st.text_input("Gemini API Key", type="password")
    submitted = st.form_submit_button("Save Keys")

if not all([openai_key, anthropic_key, gemini_key]):
    st.warning("Please enter all API keys to continue.")
    st.stop()

user_input = st.text_area("Enter your problem/task for Hydra to solve:")

if st.button("Run Hydra") and user_input:
    openai_agent = OpenAIAgent(openai_key)
    anthropic_agent = AnthropicAgent(anthropic_key)
    gemini_agent = GeminiAgent(gemini_key)

    orchestrator = Orchestrator(openai_agent, anthropic_agent, gemini_agent)
    result = orchestrator.run(user_input)

    st.header("🧠 Multi-Agent Responses")
    for stage, output in result.items():
        st.subheader(stage)
        st.write(output)
