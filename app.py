import streamlit as st
import uuid
import asyncio
from typing import Dict, Any, Optional

from models.openai_agent import OpenAIAgent
from models.anthropic_agent import AnthropicAgent
from models.gemini_agent import GeminiAgent
from orchestrator import Orchestrator
from memory.memory_manager import MemoryManager

st.set_page_config(page_title="Hydra v3 - AI Orchestrator", layout="wide")
st.title("🧠 Hydra v3 - Claude-Orchestrated Multi-Agent System")

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Keys
    with st.expander("API Keys", expanded=True):
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai")
        anthropic_key = st.text_input("Anthropic API Key", type="password", key="anthropic") 
        gemini_key = st.text_input("Gemini API Key", type="password", key="gemini")
    
    # Model Selection
    with st.expander("Model Configuration"):
        primary_model = st.selectbox(
            "Primary Orchestrator",
            ["claude-4", "gpt-4o", "gemini-2.5-flash", "gemini-2.5-pro"],
            index=0,
            help="The main model that orchestrates and performs work"
        )
        
        auto_consultation = st.checkbox(
            "Auto Consultation Mode", 
            value=True,
            help="Let primary model decide when to consult other models"
        )
        
        available_consultants = st.multiselect(
            "Available Consultant Models",
            ["gpt-4o", "gemini-2.5-pro", "gemini-2.5-flash"],
            default=["gpt-4o", "gemini-2.5-pro"],
            help="Models available for consultation"
        )

# Main interface
if not all([openai_key, anthropic_key, gemini_key]):
    st.warning("⚠️ Please enter all API keys in the sidebar to continue.")
    st.stop()

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# User input
user_input = st.text_area(
    "What do you need help with?",
    placeholder="Examples:\n• Write a horror story intro\n• Debug this authentication code\n• Analyze the architecture of my React app\n• Review this code for security issues",
    height=120
)

# Advanced options
with st.expander("⚙️ Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        max_consultations = st.slider(
            "Max Consultations per Task", 
            min_value=0, 
            max_value=5, 
            value=3,
            help="Maximum number of other models to consult"
        )
        
        thinking_depth = st.select_slider(
            "Thinking Depth",
            options=["minimal", "low", "medium", "high", "max"],
            value="medium",
            help="How deeply consultant models should analyze"
        )
    
    with col2:
        enable_web_search = st.checkbox(
            "Enable Web Search", 
            value=True,
            help="Allow models to suggest web searches for current info"
        )
        
        save_conversations = st.checkbox(
            "Save Conversation History", 
            value=True,
            help="Persist conversations across sessions"
        )

# Execution
if st.button("🚀 Execute", type="primary") and user_input:
    
    # Initialize agents with upgraded Gemini
    agents = {
        "openai": OpenAIAgent(openai_key),
        "anthropic": AnthropicAgent(anthropic_key), 
        "gemini": GeminiAgent(gemini_key, model_variant="2.5-flash")  # Default to 2.5 Flash
    }
    
    # Initialize memory manager
    memory = MemoryManager() if save_conversations else None
    
    # Initialize orchestrator with new configuration
    orchestrator = Orchestrator(
        agents=agents,
        primary_model=primary_model,
        available_consultants=available_consultants,
        auto_consultation=auto_consultation,
        max_consultations=max_consultations,
        thinking_depth=thinking_depth,
        enable_web_search=enable_web_search,
        memory=memory
    )
    
    # Create containers for real-time updates
    status_container = st.container()
    progress_container = st.container()
    results_container = st.container()
    
    async def run_task():
        with status_container:
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
        
        def update_status(message: str, progress: float = None):
            status_placeholder.info(f"🔄 {message}")
            if progress is not None:
                progress_bar.progress(progress)
        
        # Execute the task
        result = await orchestrator.execute(
            user_input, 
            session_id=st.session_state.session_id,
            progress_callback=update_status
        )
        
        # Display results
        with results_container:
            st.header("🎯 Result")
            
            # Main output
            if result.get("primary_output"):
                st.markdown("### Primary Output")
                st.write(result["primary_output"])
            
            # Show consultation details if any occurred
            if result.get("consultations"):
                with st.expander("🤝 Consultation Details"):
                    for i, consultation in enumerate(result["consultations"], 1):
                        st.markdown(f"**Consultation {i}: {consultation['model']} - {consultation['tool']}**")
                        st.write(consultation["purpose"])
                        if consultation.get("key_insights"):
                            st.markdown("*Key Insights:*")
                            st.write(consultation["key_insights"])
                        st.divider()
            
            # Show execution metrics
            if result.get("metrics"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Time", f"{result['metrics']['total_time']:.1f}s")
                with col2:
                    st.metric("Consultations Used", result['metrics']['consultations_count'])
                with col3:
                    st.metric("Primary Model Tokens", result['metrics']['primary_tokens'])
                with col4:
                    st.metric("Total Tokens", result['metrics']['total_tokens'])
        
        # Update conversation history
        st.session_state.conversation_history.append({
            "user_input": user_input,
            "result": result,
            "timestamp": result.get("timestamp")
        })
        
        # Clear status
        status_placeholder.success("✅ Task completed successfully!")
        progress_bar.progress(1.0)
    
    # Run the async task
    asyncio.run(run_task())

# Conversation History
if st.session_state.conversation_history:
    with st.expander(f"📜 Conversation History ({len(st.session_state.conversation_history)} items)"):
        for i, item in enumerate(reversed(st.session_state.conversation_history), 1):
            st.markdown(f"**Query {len(st.session_state.conversation_history) - i + 1}:** {item['user_input'][:100]}...")
            if item['result'].get('metrics'):
                st.caption(f"Consultations: {item['result']['metrics']['consultations_count']} | "
                          f"Time: {item['result']['metrics']['total_time']:.1f}s")
            st.divider()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    Hydra v3 - Claude-Orchestrated Multi-Agent System<br>
    🧠 One Primary Mind, Multiple Expert Consultants
    </div>
    """, 
    unsafe_allow_html=True
)
