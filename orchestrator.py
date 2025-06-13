import asyncio
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from tools_system_complete import WebSearchTool, ConsultationEngine, ConsultationRequest, ToolRegistry
from model_selector import ModelSelector


class TaskComplexity(Enum):
    SIMPLE = "simple"           # Direct execution, no consultation needed
    MODERATE = "moderate"       # May benefit from 1-2 consultations
    COMPLEX = "complex"         # Likely needs multiple consultations
    SPECIALIZED = "specialized" # Requires domain expert consultation


@dataclass
class ConsultationRequest:
    purpose: str
    model: str
    tool: str
    context: Dict[str, Any]
    priority: int = 1


@dataclass
class ConsultationResult:
    model: str
    tool: str
    purpose: str
    output: str
    key_insights: str
    execution_time: float
    tokens_used: int


class Orchestrator:
    """
    Claude-centric orchestrator - ENHANCED to force multi-model consultation
    """
    
    def __init__(
        self,
        agents: Dict[str, Any],
        primary_model: str = "claude-4",
        available_consultants: List[str] = None,
        auto_consultation: bool = True,
        max_consultations: int = 3,
        thinking_depth: str = "medium",
        enable_web_search: bool = True,
        memory = None
    ):
        self.agents = agents
        self.primary_model = primary_model
        self.available_consultants = available_consultants or ["gpt-4o", "gemini-2.5-pro"]
        self.auto_consultation = auto_consultation
        self.max_consultations = max_consultations
        self.thinking_depth = thinking_depth
        self.enable_web_search = enable_web_search
        self.memory = memory
        
        # Initialize consultation system
        self.consultation_engine = ConsultationEngine(agents)
        self.tool_registry = ToolRegistry()
        self.model_selector = ModelSelector()
        
        # File handling
        self.uploaded_file_paths = []
        
        # Execution state
        self.consultation_count = 0
        self.start_time = None
        self.primary_tokens = 0
        self.total_tokens = 0

    async def execute(
        self, 
        user_input: str, 
        session_id: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Main execution method - ALWAYS uses consultations for non-trivial tasks
        """
        self.start_time = time.time()
        self.consultation_count = 0
        self.primary_tokens = 0
        self.total_tokens = 0
        
        if progress_callback:
            progress_callback("Analyzing task complexity...", 0.1)
        
        # Check if this is a request for current information (time/date only)
        if self._is_simple_time_query(user_input):
            if progress_callback:
                progress_callback("Handling simple time query...", 0.3)
            
            web_result = await self._handle_web_search(user_input)
            if web_result:
                if progress_callback:
                    progress_callback("Complete!", 1.0)
                
                return {
                    "primary_output": web_result,
                    "consultations": [],
                    "metrics": {
                        "total_time": time.time() - self.start_time,
                        "consultations_count": 0,
                        "primary_tokens": 0,
                        "total_tokens": 0,
                        "task_complexity": "simple_time_query"
                    },
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "session_id": session_id
                }
        
        # For ALL other tasks, use multi-model consultation
        if progress_callback:
            progress_callback("Generating consultations for multi-perspective analysis...", 0.2)
        
        # Force consultations for almost everything
        consultations = await self._force_consultations(user_input, progress_callback)
        
        if progress_callback:
            progress_callback("Primary model synthesizing insights...", 0.7)
        
        # Primary model synthesizes with consultation input
        primary_agent = self._get_agent(self.primary_model)
        primary_prompt = self._build_synthesis_prompt(user_input, consultations)
        primary_response = await primary_agent.chat(primary_prompt)
        
        # Calculate tokens
        self.primary_tokens = self._estimate_tokens(primary_prompt + primary_response)
        self.total_tokens = self.primary_tokens
        
        for consultation in consultations:
            self.total_tokens += consultation.tokens_used
        
        if progress_callback:
            progress_callback("Complete!", 1.0)
        
        # Prepare execution result
        execution_time = time.time() - self.start_time
        
        result = {
            "primary_output": primary_response,
            "consultations": [
                {
                    "model": c.model,
                    "tool": c.tool,
                    "purpose": c.purpose,
                    "key_insights": c.key_insights
                }
                for c in consultations
            ],
            "metrics": {
                "total_time": execution_time,
                "consultations_count": self.consultation_count,
                "primary_tokens": self.primary_tokens,
                "total_tokens": self.total_tokens,
                "task_complexity": "multi_model_consultation"
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id
        }
        
        # Save to memory if enabled
        if self.memory:
            self.memory.save_session(session_id, result)
        
        return result

    async def _force_consultations(self, user_input: str, progress_callback: Optional[Callable] = None) -> List[ConsultationResult]:
        """Force consultations for better multi-perspective analysis"""
        consultations = []
        user_lower = user_input.lower()
        
        # Always get at least 2 consultations for any substantial task
        consultation_requests = []
        
        # Editorial/Writing tasks - specialized consultations
        if any(word in user_lower for word in ['edit', 'manuscript', 'chapter', 'story', 'writing', 'prose', 'narrative', 'book', 'novel', 'feedback']):
            consultation_requests = [
                ConsultationRequest(
                    purpose="Deep literary and editorial analysis",
                    model="gemini-2.5-pro",
                    tool="thinkdeep",
                    context={
                        "user_input": user_input,
                        "thinking_mode": "high",
                        "specialty": "editorial_analysis"
                    }
                ),
                ConsultationRequest(
                    purpose="Alternative writing perspective and craft analysis",
                    model="gpt-4o",
                    tool="analyze",
                    context={
                        "user_input": user_input,
                        "specialty": "writing_craft"
                    }
                )
            ]
        
        # Technical tasks
        elif any(word in user_lower for word in ['code', 'debug', 'technical', 'programming', 'software']):
            consultation_requests = [
                ConsultationRequest(
                    purpose="Technical analysis and code review",
                    model="gpt-4o",
                    tool="codereview",
                    context={"user_input": user_input}
                ),
                ConsultationRequest(
                    purpose="Deep technical thinking and architecture",
                    model="gemini-2.5-pro",
                    tool="thinkdeep",
                    context={
                        "user_input": user_input,
                        "thinking_mode": "max"
                    }
                )
            ]
        
        # Analysis tasks
        elif any(word in user_lower for word in ['analyze', 'review', 'assess', 'evaluate', 'explain']):
            consultation_requests = [
                ConsultationRequest(
                    purpose="Comprehensive analytical thinking",
                    model="gemini-2.5-pro",
                    tool="thinkdeep",
                    context={
                        "user_input": user_input,
                        "thinking_mode": "high"
                    }
                ),
                ConsultationRequest(
                    purpose="Quick analytical insights",
                    model="gemini-2.5-flash",
                    tool="analyze",
                    context={
                        "user_input": user_input,
                        "thinking_mode": "medium"
                    }
                )
            ]
        
        # General tasks - still get multiple perspectives
        else:
            consultation_requests = [
                ConsultationRequest(
                    purpose="Primary analytical perspective",
                    model="gemini-2.5-pro",
                    tool="analyze",
                    context={
                        "user_input": user_input,
                        "thinking_mode": "medium"
                    }
                ),
                ConsultationRequest(
                    purpose="Alternative analytical perspective",
                    model="gpt-4o",
                    tool="analyze",
                    context={"user_input": user_input}
                )
            ]
        
        # Execute consultations
        for i, req in enumerate(consultation_requests[:self.max_consultations]):
            if progress_callback:
                progress_percent = 0.2 + (0.5 * (i / len(consultation_requests)))
                progress_callback(f"Consulting {req.model} for {req.purpose}...", progress_percent)
            
            consultation_result = await self.consultation_engine.execute_consultation(req)
            consultations.append(consultation_result)
            self.consultation_count += 1
        
        return consultations

    def _build_synthesis_prompt(self, user_input: str, consultations: List[ConsultationResult]) -> str:
        """Build prompt for primary model to synthesize consultation results"""
        
        consultation_insights = ""
        if consultations:
            consultation_insights = "\n\n=== SPECIALIST CONSULTATIONS ===\n"
            for i, consultation in enumerate(consultations, 1):
                consultation_insights += f"\n**Consultation {i}: {consultation.model} ({consultation.tool})**\n"
                consultation_insights += f"Purpose: {consultation.purpose}\n"
                consultation_insights += f"Analysis: {consultation.output}\n"
                consultation_insights += f"Key Insights: {consultation.key_insights}\n"
                consultation_insights += "---\n"
        
        synthesis_prompt = f"""
You are the primary orchestrator synthesizing insights from multiple AI specialist consultations.

USER REQUEST: {user_input}

{consultation_insights}

TASK: Provide a comprehensive response that:
1. Synthesizes the specialist insights
2. Adds your own expertise and perspective
3. Directly addresses the user's request
4. Acknowledges different viewpoints when valuable
5. Delivers actionable, practical guidance

Your synthesized response:
"""
        
        return synthesis_prompt

    def _is_simple_time_query(self, user_input: str) -> bool:
        """Check if this is ONLY asking for current time/date (not other current info)"""
        user_lower = user_input.lower().strip()
        
        # Only very simple time/date queries
        simple_time_queries = [
            'what time is it',
            'current time',
            'what day is it',
            'today\'s date',
            'what date is it'
        ]
        
        return any(query in user_lower for query in simple_time_queries) and len(user_input) < 50

    def _get_agent(self, model_name: str):
        """Get the appropriate agent for a model"""
        model_map = {
            "claude-4": "anthropic",
            "gpt-4o": "openai", 
            "gemini-2.5-pro": "gemini",
            "gemini-2.5-flash": "gemini"
        }
        
        agent_type = model_map.get(model_name, "anthropic")
        return self.agents[agent_type]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return int(len(text.split()) * 1.3)

    async def _handle_web_search(self, user_input: str) -> str:
        """Handle simple time queries"""
        try:
            search_request = ConsultationRequest(
                purpose=f"Get current time/date: {user_input}",
                model="gemini-2.5-flash",
                tool="websearch",
                context={
                    "user_input": user_input,
                    "search_query": user_input,
                    "approach": "Direct time query"
                }
            )
            
            result = await self.consultation_engine.execute_consultation(search_request)
            return result.output
                    
        except Exception as e:
            return f"I'm unable to access real-time information at the moment. Error: {str(e)}"
