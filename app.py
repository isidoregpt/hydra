import streamlit as st
import uuid
import asyncio
from typing import Dict, Any, Optional

from models.openai_agent import OpenAIAgent
from models.anthropic_agent import AnthropicAgent
from models.gemini_agent import GeminiAgent
from orchestrator import Orchestrator
from memory.memory_manager import MemoryManager
from file_manager import FileManager, FileAnalysisInterface

st.set_page_config(page_title="Hydra v3 - AI Orchestrator", layout="wide")
st.title("🧠 Hydra v3 - Claude Opus 4 Orchestrated Multi-Agent System")

# Instructions and About sections for end users
col1, col2 = st.columns(2)

with col1:
    with st.expander("📋 Instructions - How to Use This App", expanded=False):
        st.markdown("""
        ## **Simple 4-Step Process**
        
        ### **Step 1: Enter Your API Keys** 🔑
        - Click the **sidebar** (arrow on the left)
        - Enter your API keys in the "API Keys" section
        - **Need keys?** Get them here:
          - **Anthropic**: [console.anthropic.com](https://console.anthropic.com/settings/keys) (for Claude)
          - **OpenAI**: [platform.openai.com](https://platform.openai.com/account/api-keys) (for GPT)
          - **Google**: [aistudio.google.com](https://aistudio.google.com/app/apikey) (for Gemini)
        
        ### **Step 2: Upload Files (Optional)** 📁
        - **Drag and drop** files into the upload area
        - Supports: documents, code, spreadsheets, PDFs, ZIP folders
        - **Examples**: manuscripts, reports, code projects, data files
        
        ### **Step 3: Ask Your Question** ❓
        - Type what you want help with in the big text box
        - **Examples**:
          - *"Review my manuscript and give me editorial feedback"*
          - *"Analyze this code for bugs and improvements"*
          - *"Help me understand this data and find insights"*
          - *"Explain this document in simple terms"*
        
        ### **Step 4: Click Execute and Wait** 🚀
        - Hit the **blue "Execute" button**
        - Watch the progress bar as multiple AI models work together
        - **Be patient** - quality takes time (usually 3-5 minutes)
        
        ---
        
        ## **💡 Pro Tips**
        
        **✅ What Works Best:**
        - Be specific about what you want
        - Upload relevant files for context
        - Ask for detailed analysis or feedback
        - Use for complex tasks that benefit from multiple perspectives
        
        **❌ What to Avoid:**
        - Very simple questions (like "what is 2+2?")
        - Requests for illegal or harmful content
        - Extremely long documents (break them into chunks)
        
        **🎯 Perfect Use Cases:**
        - **Writers**: Manuscript reviews, editing suggestions
        - **Developers**: Code reviews, debugging help  
        - **Students**: Essay feedback, research analysis
        - **Business**: Report analysis, decision support
        - **Researchers**: Data analysis, literature reviews
        
        ---
        
        ## **🔧 Settings (Advanced)**
        
        **In the sidebar, you can:**
        - **Change the primary AI** (Claude Opus 4 recommended)
        - **Add/remove consultant AIs** (more = better analysis)
        - **Adjust thinking depth** (higher = more thorough)
        - **Enable web search** (for current information)
        
        **💡 Tip**: Default settings work great for most people!
        """)

with col2:
    with st.expander("ℹ️ About - What Makes This Special", expanded=False):
        st.markdown("""
        ## **What Is Hydra v3?**
        
        Imagine having a **team of expert consultants** who:
        - Each review your work independently
        - Discuss and debate their findings
        - Build consensus on the best advice
        - Give you their collective wisdom
        
        That's exactly what Hydra v3 does, but with **AI models** instead of humans!
        
        ---
        
        ## **🤖 Your AI Team**
        
        **Meet Your AI Consultants:**
        
        **🧠 Claude Opus 4** *(Team Leader)*
        - The newest, most intelligent AI from Anthropic
        - Coordinates the entire team and makes final decisions
        - Best at complex reasoning and synthesis
        
        **⚡ Claude Sonnet 4** *(Performance Specialist)*  
        - High-speed analysis with exceptional reasoning
        - Great at detailed reviews and critiques
        
        **🎯 Gemini 2.5 Pro** *(Deep Thinker)*
        - Google's most powerful AI with "thinking modes"
        - Excels at complex analysis and creative solutions
        
        **🔧 GPT-4o** *(Problem Solver)*
        - OpenAI's versatile AI with strong general intelligence
        - Excellent at practical advice and debugging
        
        ---
        
        ## **🔄 How They Work Together**
        
        ### **Round 1: Independent Analysis**
        Each AI looks at your request separately and gives their own analysis.
        
        ### **Round 2: Peer Review**
        The AIs read each other's work and provide constructive criticism.
        
        ### **Round 3: Building Agreement** 
        They work together to resolve disagreements and find consensus.
        
        ### **Final: Team Synthesis**
        Claude Opus 4 combines everyone's insights into one comprehensive response.
        
        ---
        
        ## **🌟 Why This Is Revolutionary**
        
        **Traditional AI Apps:**
        - Ask one AI → Get one perspective
        - Limited by that AI's particular strengths/weaknesses
        - No discussion or refinement of ideas
        
        **Hydra v3:**
        - Ask multiple AIs → Get team analysis
        - AIs actually collaborate and improve each other's work
        - Final answer is better than any single AI could provide
        
        **Real Examples:**
        - **Single AI**: *"Your writing needs work"*
        - **Hydra v3**: *"After collaborative analysis, we found 3 key areas for improvement, with specific examples and step-by-step solutions that all AIs agreed upon"*
        
        ---
        
        ## **📊 What You'll See**
        
        **While It's Working:**
        - Progress bar showing which AI is working
        - Updates like "Round 1: GPT-4o analyzing..." 
        - Total process usually takes 3-5 minutes
        
        **When It's Done:**
        - **Main Result**: The team's final collaborative answer
        - **Consultation Details**: What each AI contributed
        - **Metrics**: How much work was done (tokens, time, consultations)
        
        ---
        
        ## **🎯 Perfect For:**
        
        **📝 Writers & Authors**
        - Get comprehensive manuscript feedback
        - Multiple perspectives on your writing
        - Editorial advice from different angles
        
        **💻 Developers & Programmers**
        - Code reviews from multiple AI experts
        - Debugging help with different approaches
        - Architecture advice and best practices
        
        **🎓 Students & Researchers**
        - Essay feedback and improvement suggestions
        - Research analysis from multiple viewpoints
        - Help understanding complex topics
        
        **💼 Business Professionals**
        - Document analysis and insights
        - Decision support with multiple perspectives
        - Strategy development and planning
        
        ---
        
        ## **🚀 The Bottom Line**
        
        Hydra v3 is like having a **panel of AI experts** work together on your problem. 
        
        Instead of getting one AI's opinion, you get the **collective intelligence** of multiple state-of-the-art AIs who have actually collaborated, discussed, and agreed on the best solution for you.
        
        **It's not just multiple AIs - it's AIs working as a team!** 🤖🤝🤖🤝🤖
        """)

st.markdown("---")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "file_manager" not in st.session_state:
    st.session_state.file_manager = FileManager(st.session_state.session_id)

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
            ["claude-opus-4", "claude-sonnet-4", "gpt-4o", "gemini-2.5-pro"],
            index=0,
            help="The main model that orchestrates and performs work. Claude Opus 4 recommended for best results."
        )
        
        auto_consultation = st.checkbox(
            "Multi-Round Consultation", 
            value=True,
            help="Enable collaborative multi-round consultation with cross-review and consensus building"
        )
        
        available_consultants = st.multiselect(
            "Available Consultant Models",
            ["claude-sonnet-4", "gpt-4o", "gemini-2.5-pro", "gemini-2.5-flash"],
            default=["claude-sonnet-4", "gpt-4o", "gemini-2.5-pro"],
            help="Models available for collaborative consultation"
        )

# Main interface - Two columns
col1, col2 = st.columns([2, 1])

with col1:
    # File Upload Section
    st.header("📁 File Upload")
    
    uploaded_files = st.file_uploader(
        "Upload files or folders (ZIP) for analysis",
        accept_multiple_files=True,
        type=['py', 'js', 'jsx', 'ts', 'tsx', 'java', 'cpp', 'c', 'h', 'css', 'html', 'php', 'rb', 'go', 'rs', 'swift', 'kt',
              'txt', 'md', 'pdf', 'doc', 'docx', 'rtf',
              'csv', 'json', 'xml', 'yaml', 'yml', 'sql',
              'env', 'ini', 'conf', 'config', 'toml', 'zip'],
        help="Upload individual files or ZIP archives containing folders. Supports code, documents, data files, and configuration files."
    )
    
    # Process uploaded files
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            results = st.session_state.file_manager.process_uploaded_files(uploaded_files)
        
        if results["errors"]:
            for error in results["errors"]:
                st.error(error)
        
        if results["total_files"] > 0:
            st.success(f"✅ Successfully processed {results['total_files']} files")
            
            # Show file summary
            with st.expander("📊 File Summary", expanded=True):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Total Files", results["total_files"])
                    st.metric("Total Size", FileManager.format_file_size(results["total_size"]))
                
                with col_b:
                    if results["file_types"]:
                        st.write("**File Types:**")
                        for file_type, count in results["file_types"].items():
                            st.write(f"• {file_type.title()}: {count}")

with col2:
    # File Analysis Interface
    if st.session_state.file_manager.processed_files:
        st.header("🔍 Quick Analysis")
        
        file_analysis = FileAnalysisInterface(st.session_state.file_manager)
        file_summary = st.session_state.file_manager.get_file_summary()
        suggestions = file_analysis.suggest_analysis_approaches(file_summary)
        
        st.write("**Suggested Analyses:**")
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggest_{suggestion[:10]}"):
                # Auto-fill the query with the suggestion
                st.session_state.auto_query = suggestion.split(": ", 1)[-1] if ": " in suggestion else suggestion

# Main query interface
if not all([openai_key, anthropic_key, gemini_key]):
    st.warning("⚠️ Please enter all API keys in the sidebar to continue.")
    st.stop()

# User input with file context
user_input = st.text_area(
    "What would you like me to analyze or help you with?",
    value=st.session_state.get("auto_query", ""),
    placeholder="Examples:\n• Analyze the uploaded code for security issues\n• Review the data structure in my CSV files\n• Explain what this codebase does\n• Provide editorial feedback on my manuscript\n• Debug this code issue",
    height=120,
    key="main_query"
)

# Clear auto query after use
if "auto_query" in st.session_state:
    del st.session_state.auto_query

# Advanced options
with st.expander("⚙️ Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        max_consultations = st.slider(
            "Max Consultations per Round", 
            min_value=1, 
            max_value=6, 
            value=4,
            help="Maximum number of models to consult per round"
        )
        
        thinking_depth = st.select_slider(
            "Thinking Depth",
            options=["minimal", "low", "medium", "high", "max"],
            value="high",
            help="How deeply models should think (applies to Gemini 2.5 models)"
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
    
    # Initialize agents with Claude Opus 4 as primary
    agents = {
        "openai": OpenAIAgent(openai_key),
        "anthropic": AnthropicAgent(anthropic_key, model_variant="opus-4"),  # Claude Opus 4
        "gemini": GeminiAgent(gemini_key, model_variant="2.5-flash")
    }
    
    # Initialize memory manager
    memory = MemoryManager() if save_conversations else None
    
    # Initialize orchestrator with Claude Opus 4 configuration
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
    results_container = st.container()
    
    async def run_task():
        with status_container:
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
        
        def update_status(message: str, progress: float = None):
            status_placeholder.info(f"🔄 {message}")
            if progress is not None:
                progress_bar.progress(progress)
        
        # Enhanced user input with file context
        file_analysis = FileAnalysisInterface(st.session_state.file_manager)
        enhanced_query = file_analysis.create_file_context_prompt(user_input)
        
        # Add file paths to orchestrator context if files are uploaded
        if st.session_state.file_manager.processed_files:
            orchestrator.uploaded_file_paths = st.session_state.file_manager.get_analysis_ready_paths()
        
        # Execute the task
        result = await orchestrator.execute(
            enhanced_query, 
            session_id=st.session_state.session_id,
            progress_callback=update_status
        )
        
        # Display results
        with results_container:
            st.header("🎯 Result")
            
            # Show file context if files were used
            if st.session_state.file_manager.processed_files:
                file_summary = st.session_state.file_manager.get_file_summary()
                st.info(f"📁 Analysis included {file_summary['total_files']} uploaded files ({FileManager.format_file_size(file_summary['total_size'])})")
            
            # Main output
            if result.get("primary_output"):
                st.markdown("### Primary Output")
                st.write(result["primary_output"])
            
            # Show collaboration rounds if available
            if result.get("collaboration_rounds"):
                with st.expander("🤝 Multi-Round Collaboration Details"):
                    rounds = result["collaboration_rounds"]
                    if isinstance(rounds, dict) and not rounds.get("simple_query"):
                        st.write(f"**Round 1 - Independent Analysis:** {rounds.get('round_1_independent', 0)} consultations")
                        st.write(f"**Round 2 - Cross Review:** {rounds.get('round_2_cross_review', 0)} consultations")
                        st.write(f"**Round 3 - Consensus Building:** {rounds.get('round_3_consensus', 0)} consultations")
            
            # Show consultation details if any occurred
            if result.get("consultations"):
                with st.expander("🤖 Individual Consultation Details"):
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
                    st.metric("Primary Model Tokens", f"{result['metrics']['primary_tokens']:,}")
                with col4:
                    st.metric("Total Tokens", f"{result['metrics']['total_tokens']:,.0f}")
        
        # Update conversation history
        st.session_state.conversation_history.append({
            "user_input": user_input,
            "result": result,
            "timestamp": result.get("timestamp"),
            "files_included": len(st.session_state.file_manager.processed_files) > 0
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
            files_badge = " 📁" if item.get('files_included') else ""
            st.markdown(f"**Query {len(st.session_state.conversation_history) - i + 1}:**{files_badge} {item['user_input'][:100]}...")
            if item['result'].get('metrics'):
                st.caption(f"Consultations: {item['result']['metrics']['consultations_count']} | "
                          f"Time: {item['result']['metrics']['total_time']:.1f}s | "
                          f"Tokens: {item['result']['metrics']['total_tokens']:,.0f}")
            st.divider()

# File Management
if st.session_state.file_manager.processed_files:
    with st.expander("🗂️ File Management"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            file_summary = st.session_state.file_manager.get_file_summary()
            st.write(f"**Files ready for analysis:** {file_summary['total_files']}")
            
        with col2:
            if st.button("🗑️ Clear Files", help="Remove all uploaded files"):
                st.session_state.file_manager.cleanup()
                st.session_state.file_manager = FileManager(st.session_state.session_id)
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    Hydra v3 - Claude Opus 4 Orchestrated Multi-Agent System<br>
    🧠 Claude Opus 4 + Claude Sonnet 4 + Gemini 2.5 Pro + GPT-4o + Multi-Round Collaboration 🚀
    </div>
    """, 
    unsafe_allow_html=True
)
