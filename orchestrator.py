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
    Claude-centric orchestrator that decides when and how to use other models
    as specialized consultants rather than equals in a democracy.
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
        # Updated default consultants to use official Gemini 2.5 models
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
        Main execution method - Claude analyzes the task and decides how to approach it
        """
        self.start_time = time.time()
        self.consultation_count = 0
        self.primary_tokens = 0
        self.total_tokens = 0
        
        if progress_callback:
            progress_callback("Analyzing task complexity...", 0.1)
        
        # Check if this is a request for current information
        if self._needs_current_info(user_input):
            if progress_callback:
                progress_callback("Searching for current information...", 0.3)
            
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
                        "task_complexity": "simple_web_search"
                    },
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "session_id": session_id
                }
        
        # Continue with normal task processing
        # Step 1: Classify the task complexity
        task_complexity = await self._classify_task(user_input)
        
        # Step 2: Enhanced consultation strategy - be more aggressive about consulting
        consultations = []
        
        # Always get consultations for moderate+ complexity tasks when auto_consultation is enabled
        if self.auto_consultation and task_complexity != TaskComplexity.SIMPLE:
            if progress_callback:
                progress_callback("Identifying consultation opportunities...", 0.2)
            
            consultation_requests = self._generate_strategic_consultations(user_input, task_complexity)
            
            # Execute consultations BEFORE primary analysis for better input
            for req in consultation_requests[:self.max_consultations]:
                if progress_callback:
                    progress_percent = 0.2 + (0.4 * (len(consultations) / max(1, len(consultation_requests))))
                    progress_callback(f"Consulting {req.model} for {req.purpose}...", progress_percent)
                
                consultation_result = await self.consultation_engine.execute_consultation(req)
                consultations.append(consultation_result)
                self.consultation_count += 1
        
        # Step 3: Primary model analyzes with consultation input
        primary_agent = self._get_agent(self.primary_model)
        
        if progress_callback:
            progress_callback("Primary model analyzing with consultation input...", 0.6)
        
        primary_prompt = self._build_enhanced_primary_prompt(user_input, task_complexity, consultations)
        primary_response = await primary_agent.chat(primary_prompt)
        self.primary_tokens += self._estimate_tokens(primary_prompt + primary_response)
        self.total_tokens += self.primary_tokens
        
        if progress_callback:
            progress_callback("Synthesizing final response...", 0.8)
        
        # Step 4: Add consultation token counts
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
                "task_complexity": task_complexity.value
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id
        }
        
        # Save to memory if enabled
        if self.memory:
            self.memory.save_session(session_id, result)
        
        return result

    async def _classify_task(self, user_input: str) -> TaskComplexity:
        """Enhanced task complexity classification"""
        user_lower = user_input.lower()
        
        # Simple creative tasks (only very basic ones)
        if any(phrase in user_lower for phrase in ['write a', 'create a', 'generate a']) and \
           any(phrase in user_lower for phrase in ['short story', 'quick poem', 'simple intro']) and \
           len(user_input) < 80:
            return TaskComplexity.SIMPLE
        
        # Specialized domains - always complex
        if any(phrase in user_lower for phrase in ['security', 'cryptography', 'machine learning', 'architecture', 'editorial', 'manuscript', 'book', 'novel']):
            return TaskComplexity.SPECIALIZED
        
        # Complex indicators
        if any(phrase in user_lower for phrase in ['complex', 'comprehensive', 'enterprise', 'scalable', 'review', 'analyze', 'feedback', 'edit']):
            return TaskComplexity.COMPLEX
        
        # Technical tasks - moderate to complex
        if any(phrase in user_lower for phrase in ['debug', 'optimize', 'code', 'technical']):
            return TaskComplexity.MODERATE
        
        # Content analysis/creation tasks
        if any(phrase in user_lower for phrase in ['chapter', 'story', 'writing', 'prose', 'narrative', 'character']):
            return TaskComplexity.COMPLEX
        
        # Default to moderate to encourage more consultation
        return TaskComplexity.MODERATE

    def _generate_strategic_consultations(self, user_input: str, complexity: TaskComplexity) -> List[ConsultationRequest]:
        """Generate strategic consultations based on task analysis"""
        consultations = []
        user_lower = user_input.lower()
        context_keywords = user_input.split()
        
        # Editorial/Writing tasks - always get multiple perspectives
        if any(word in user_lower for word in ['edit', 'manuscript', 'chapter', 'story', 'writing', 'prose', 'narrative']):
            consultations.extend([
                ConsultationRequest(
                    purpose="Editorial analysis and writing quality assessment",
                    model="gemini-2.5-pro",
                    tool="analyze",
                    context={
                        "user_input": user_input,
                        "thinking_mode": "high",
                        "complexity": complexity.value,
                        "keywords": context_keywords,
                        "specialty": "literary_analysis"
                    }
                ),
                ConsultationRequest(
                    purpose="Alternative perspective on writing style and structure",
                    model="gpt-4o",
                    tool="analyze",
                    context={
                        "user_input": user_input,
                        "complexity": complexity.value,
                        "keywords": context_keywords,
                        "specialty": "writing_craft"
                    }
                )
            ])
        
        # Code/Technical tasks
        elif any(word in user_lower for word in ['code', 'debug', 'review', 'programming', 'technical']):
            consultations.extend([
                ConsultationRequest(
                    purpose="Technical analysis and code review",
                    model="gpt-4o",
                    tool="codereview",
                    context={
                        "user_input": user_input,
                        "complexity": complexity.value,
                        "keywords": context_keywords
                    }
                ),
                ConsultationRequest(
                    purpose="Deep architectural analysis",
                    model="gemini-2.5-pro",
                    tool="thinkdeep",
                    context={
                        "user_input": user_input,
                        "thinking_mode": "max",
                        "complexity": complexity.value,
                        "keywords": context_keywords
                    }
                )
            ])
        
        # Analysis tasks
        elif any(word in user_lower for word in ['analyze', 'review', 'assess', 'evaluate']):
            consultations.extend([
                ConsultationRequest(
                    purpose="Comprehensive analysis from multiple angles",
                    model="gemini-2.5-pro",
                    tool="thinkdeep",
                    context={
                        "user_input": user_input,
                        "thinking_mode": "high",
                        "complexity": complexity.value,
                        "keywords": context_keywords
                    }
                ),
                ConsultationRequest(
                    purpose="Quick analytical insights and validation",
                    model="gemini-2.5-flash",
                    tool="analyze",
                    context={
                        "user_input": user_input,
                        "thinking_mode": "medium",
                        "complexity": complexity.value,
                        "keywords": context_keywords
                    }
                )
            ])
        
        # Creative/Complex tasks - get diverse perspectives
        else:
            # Use model selector for intelligent recommendations
            recommendations = self.model_selector.get_model_recommendations(user_input)
            
            for aspect, recommended_model in recommendations.items():
                if recommended_model in self.available_consultants:
                    aspect_tool_map = {
                        'security': 'codereview',
                        'debugging': 'debug', 
                        'architecture': 'thinkdeep',
                        'performance': 'analyze',
                        'thinking': 'thinkdeep',
                        'general': 'analyze'
                    }
                    
                    tool = aspect_tool_map.get(aspect, 'analyze')
                    thinking_mode = self.model_selector.select_thinking_mode(
                        recommended_model, tool, complexity.value
                    )
                    
                    consultations.append(ConsultationRequest(
                        purpose=f"{aspect} consultation using {recommended_model}",
                        model=recommended_model,
                        tool=tool,
                        context={
                            "user_input": user_input,
                            "thinking_mode": thinking_mode,
                            "complexity": complexity.value,
                            "keywords": context_keywords,
                            "file_paths": getattr(self, 'uploaded_file_paths', [])
                        }
                    ))
        
        return consultations

    def _build_enhanced_primary_prompt(self, user_input: str, complexity: TaskComplexity, consultations: List[ConsultationResult]) -> str:
        """Build enhanced prompt that incorporates consultation results"""
        
        consultation_summary = ""
        if consultations:
            consultation_summary = "\n\nCONSULTATION INSIGHTS:\n"
            for i, consultation in enumerate(consultations, 1):
                consultation_summary += f"\n{i}. {consultation.model} ({consultation.tool}) - {consultation.purpose}:\n"
                consultation_summary += f"   Key Insights: {consultation.key_insights}\n"
                consultation_summary += f"   Full Analysis: {consultation.output[:500]}{'...' if len(consultation.output) > 500 else ''}\n"
        
        enhanced_prompt = f"""
You are the primary orchestrator in a multi-agent AI system. You have received specialist consultations from other AI models and should now provide the final, comprehensive response.

USER REQUEST: {user_input}

TASK COMPLEXITY: {complexity.value}

{consultation_summary}

INSTRUCTIONS:
- Synthesize the consultation insights with your own analysis
- Provide a comprehensive, well-structured response
- Acknowledge different perspectives when relevant
- Ensure the final output directly addresses the user's request
- Build upon the specialist insights while maintaining your own voice and expertise

Your comprehensive response:
"""
        
        return enhanced_prompt

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
        return int(len(text.split()) * 1.3)  # Approximate tokens

    def _needs_current_info(self, user_input: str) -> bool:
        """Check if the request needs current/real-time information."""
        current_info_indicators = [
            'today', 'current', 'now', 'latest', 'recent', 'what time',
            'today\'s date', 'current date', 'what day', 'weather',
            'news', 'stock price', 'exchange rate', 'current status'
        ]
        
        user_lower = user_input.lower()
        return any(indicator in user_lower for indicator in current_info_indicators)

    async def _handle_web_search(self, user_input: str) -> str:
        """Handle web search for current information using the consultation engine"""
        try:
            # Create a web search consultation request
            search_request = ConsultationRequest(
                purpose=f"Search for current information: {user_input}",
                model="gemini-2.5-flash",  # Fast model for search
                tool="websearch",
                context={
                    "user_input": user_input,
                    "search_query": user_input,
                    "approach": "Direct web search consultation"
                }
            )
            
            # Execute the web search consultation
            result = await self.consultation_engine.execute_consultation(search_request)
            
            # Return the search result
            return result.output
                    
        except Exception as e:
            return f"I'm unable to access real-time information at the moment. Error: {str(e)}\n\nPlease search for '{user_input}' manually using your preferred search engine."
