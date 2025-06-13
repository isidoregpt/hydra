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
        self.available_consultants = available_consultants or ["gpt-4o", "gemini-2.0-pro"]
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
        
        # Step 2: Primary model analyzes and decides approach
        primary_agent = self._get_agent(self.primary_model)
        
        approach_prompt = self._build_approach_prompt(user_input, task_complexity)
        approach_response = await primary_agent.chat(approach_prompt)
        self.primary_tokens += self._estimate_tokens(approach_prompt + approach_response)
        
        if progress_callback:
            progress_callback("Primary analysis complete, executing approach...", 0.3)
        
        # Step 3: Execute the approach
        consultations = []
        
        # Parse approach to identify consultation needs
        consultation_requests = self._parse_consultation_needs(approach_response, user_input)
        
        # Execute consultations if needed and beneficial
        for req in consultation_requests[:self.max_consultations]:
            if progress_callback:
                progress_percent = 0.3 + (0.5 * (len(consultations) / max(1, len(consultation_requests))))
                progress_callback(f"Consulting {req.model} for {req.purpose}...", progress_percent)
            
            consultation_result = await self.consultation_engine.execute_consultation(req)
            consultations.append(consultation_result)
            self.consultation_count += 1
        
        if progress_callback:
            progress_callback("Synthesizing final response...", 0.8)
        
        # Step 4: Primary model synthesizes final output
        final_output = await self._synthesize_final_output(
            user_input, approach_response, consultations
        )
        
        if progress_callback:
            progress_callback("Complete!", 1.0)
        
        # Prepare execution result
        execution_time = time.time() - self.start_time
        
        result = {
            "primary_output": final_output,
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
        """Classify task complexity to guide orchestration approach"""
        user_lower = user_input.lower()
        
        # Simple creative tasks
        if any(phrase in user_lower for phrase in ['write a', 'create a', 'generate a']) and \
           any(phrase in user_lower for phrase in ['story', 'poem', 'intro', 'summary']):
            if len(user_input) < 100:
                return TaskComplexity.SIMPLE
        
        # Specialized domains
        if any(phrase in user_lower for phrase in ['security', 'cryptography', 'machine learning', 'architecture']):
            return TaskComplexity.SPECIALIZED
        
        # Complex indicators
        if any(phrase in user_lower for phrase in ['complex', 'comprehensive', 'enterprise', 'scalable']):
            return TaskComplexity.COMPLEX
        
        # Technical tasks
        if any(phrase in user_lower for phrase in ['debug', 'analyze', 'review', 'optimize']):
            return TaskComplexity.MODERATE
        
        # Default
        return TaskComplexity.MODERATE

    def _build_approach_prompt(self, user_input: str, complexity: TaskComplexity) -> str:
        """Build the prompt for primary model to analyze and plan approach"""
        
        base_prompt = f"""
You are the primary orchestrator in a multi-agent AI system. Your job is to:
1. Analyze the user's request and determine the best approach
2. Decide if you need consultation from specialist models 
3. Execute the primary work yourself

User Request: {user_input}

Task Complexity: {complexity.value}

Available Consultant Models:
{self._format_available_consultants()}

Available Consultation Tools:
- analyze: Deep analysis of code, architecture, or documents
- codereview: Professional code review with severity ratings  
- debug: Root cause analysis for bugs and issues
- thinkdeep: Extended reasoning for complex problems
- validate: Check work against requirements
- brainstorm: Collaborative ideation and problem-solving

Your approach should be:
- DIRECT EXECUTION for simple creative/straightforward tasks
- CONSULTATION for complex analysis, specialized domains, or validation needs
- HYBRID for tasks requiring both creative work and technical analysis

Guidelines:
- For simple creative tasks (stories, poems): Handle directly
- For technical tasks: Consider consultation for expertise
- For complex/specialized tasks: Consultation highly recommended
- Always do the primary work yourself - consultants provide input only

Analyze the request and respond with:
1. Your assessment of what's needed
2. Whether you'll handle it directly or need consultations
3. If consultations needed, specify: model, tool, and specific purpose
4. Your planned approach

Then begin executing your approach.
"""
        
        return base_prompt

    def _format_available_consultants(self) -> str:
        """Format available consultant models for the prompt"""
        descriptions = {
            "gpt-4o": "Strong reasoning, general intelligence, coding",
            "gemini-2.0-pro": "Deep analysis, architecture review, extended thinking", 
            "gemini-2.0-flash": "Fast analysis, quick validation, formatting"
        }
        
        formatted = []
        for model in self.available_consultants:
            desc = descriptions.get(model, "General purpose")
            formatted.append(f"- {model}: {desc}")
        
        return "\n".join(formatted)

    def _parse_consultation_needs(self, approach_response: str, user_input: str) -> List[ConsultationRequest]:
        """Parse the primary model's response to identify consultation requests"""
        consultation_requests = []
        
        # Look for explicit consultation requests in the response
        lines = approach_response.lower().split('\n')
        
        for line in lines:
            if any(word in line for word in ['consult', 'use', 'get', 'ask']):
                for model in self.available_consultants:
                    if model.lower() in line:
                        # Extract purpose and tool
                        purpose = self._extract_purpose_from_line(line)
                        tool = self._extract_tool_from_line(line)
                        
                        consultation_requests.append(ConsultationRequest(
                            purpose=purpose,
                            model=model,
                            tool=tool,
                            context={"user_input": user_input, "approach": approach_response}
                        ))
        
        # If auto consultation is enabled and no explicit requests, suggest based on task type
        if not consultation_requests and self.auto_consultation:
            consultation_requests.extend(self._suggest_smart_consultations(user_input, approach_response))
        
        return consultation_requests

    def _suggest_smart_consultations(self, user_input: str, approach: str) -> List[ConsultationRequest]:
        """Use ModelSelector for intelligent consultation suggestions"""
        suggestions = []
        user_lower = user_input.lower()
        context_keywords = user_input.split()
        
        # Determine task complexity
        complexity = "moderate"  # default
        if any(word in user_lower for word in ['complex', 'comprehensive', 'enterprise', 'scalable']):
            complexity = "complex"
        elif any(word in user_lower for word in ['security', 'cryptography', 'machine learning', 'architecture']):
            complexity = "specialized"
        elif any(word in user_lower for word in ['simple', 'basic', 'quick']):
            complexity = "simple"
        
        # Get model recommendations for this task
        recommendations = self.model_selector.get_model_recommendations(user_input)
        
        # Convert recommendations to consultation requests
        for aspect, recommended_model in recommendations.items():
            # Map aspects to tools
            aspect_tool_map = {
                'security': 'codereview',
                'debugging': 'debug', 
                'architecture': 'thinkdeep',
                'performance': 'analyze',
                'general': 'analyze'
            }
            
            tool = aspect_tool_map.get(aspect, 'analyze')
            
            # Only add if model is available
            if recommended_model in self.available_consultants:
                # Select appropriate thinking mode
                thinking_mode = self.model_selector.select_thinking_mode(
                    recommended_model, tool, complexity
                )
                
                suggestions.append(ConsultationRequest(
                    purpose=f"{aspect} analysis using {recommended_model}",
                    model=recommended_model,
                    tool=tool,
                    context={
                        "user_input": user_input, 
                        "approach": approach,
                        "thinking_mode": thinking_mode,
                        "complexity": complexity,
                        "keywords": context_keywords,
                        "file_paths": getattr(self, 'uploaded_file_paths', [])
                    }
                ))
        
        return suggestions[:self.max_consultations]  # Limit to max consultations

    def _extract_purpose_from_line(self, line: str) -> str:
        """Extract consultation purpose from a line of text"""
        if 'debug' in line:
            return "debugging analysis"
        elif 'review' in line:
            return "code review"
        elif 'analyze' in line:
            return "detailed analysis"
        elif 'security' in line:
            return "security analysis"
        elif 'performance' in line:
            return "performance optimization"
        else:
            return "general consultation"

    def _extract_tool_from_line(self, line: str) -> str:
        """Extract tool name from a line of text"""
        tools = ['analyze', 'codereview', 'debug', 'thinkdeep', 'validate', 'brainstorm']
        for tool in tools:
            if tool in line:
                return tool
        return 'analyze'  # default

    def _suggest_auto_consultations(self, user_input: str, approach: str) -> List[ConsultationRequest]:
        """Auto-suggest consultations based on task characteristics"""
        suggestions = []
        user_lower = user_input.lower()
        
        # Code-related tasks
        if any(word in user_lower for word in ['code', 'bug', 'debug', 'error', 'function']):
            suggestions.append(ConsultationRequest(
                purpose="code analysis and debugging",
                model="gpt-4o",
                tool="debug",
                context={"user_input": user_input, "approach": approach}
            ))
        
        # Architecture/design tasks  
        if any(word in user_lower for word in ['architecture', 'design', 'system', 'scalability']):
            suggestions.append(ConsultationRequest(
                purpose="architectural analysis",
                model="gemini-2.0-pro", 
                tool="analyze",
                context={"user_input": user_input, "approach": approach}
            ))
        
        # Security-related
        if any(word in user_lower for word in ['security', 'vulnerability', 'auth', 'permissions']):
            suggestions.append(ConsultationRequest(
                purpose="security review",
                model="gemini-2.0-pro",
                tool="codereview", 
                context={"user_input": user_input, "approach": approach}
            ))
        
        return suggestions[:2]  # Limit auto-suggestions

    async def _execute_consultation(self, request: ConsultationRequest) -> ConsultationResult:
        """Execute a single consultation with a specialist model"""
        start_time = time.time()
        
        consultant_agent = self._get_agent(request.model)
        
        # Build consultation prompt
        consultation_prompt = self._build_consultation_prompt(request)
        
        # Execute consultation
        consultation_output = await consultant_agent.chat(consultation_prompt)
        
        # Extract key insights
        key_insights = self._extract_key_insights(consultation_output)
        
        execution_time = time.time() - start_time
        tokens_used = self._estimate_tokens(consultation_prompt + consultation_output)
        self.total_tokens += tokens_used
        
        return ConsultationResult(
            model=request.model,
            tool=request.tool,
            purpose=request.purpose,
            output=consultation_output,
            key_insights=key_insights,
            execution_time=execution_time,
            tokens_used=tokens_used
        )

    def _build_consultation_prompt(self, request: ConsultationRequest) -> str:
        """Build the consultation prompt for specialist models"""
        
        base_prompt = f"""
You are a specialist consultant in a multi-agent system. Your role is to provide expert analysis in your domain.

Consultation Purpose: {request.purpose}
Tool: {request.tool}
Thinking Depth: {self.thinking_depth}

Original User Request: {request.context['user_input']}

Primary Agent's Approach: {request.context['approach']}

Provide focused, actionable insights for your specialization. Be:
- Specific and concrete
- Focused on your area of expertise  
- Actionable in your recommendations
- Aware this is consultation, not final output

Your analysis:
"""
        
        return base_prompt

    def _extract_key_insights(self, consultation_output: str) -> str:
        """Extract key insights from consultation output"""
        lines = consultation_output.split('\n')
        insights = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['key', 'important', 'critical', 'recommend', 'suggest']):
                insights.append(line)
        
        return '\n'.join(insights[:3])  # Top 3 insights

    async def _synthesize_final_output(
        self, 
        user_input: str, 
        approach: str, 
        consultations: List[ConsultationResult]
    ) -> str:
        """Primary model synthesizes consultations into final output"""
        
        primary_agent = self._get_agent(self.primary_model)
        
        consultation_summary = ""
        if consultations:
            consultation_summary = "\n\nConsultation Results:\n"
            for i, consultation in enumerate(consultations, 1):
                consultation_summary += f"\n{i}. {consultation.model} ({consultation.tool}):\n"
                consultation_summary += f"   Purpose: {consultation.purpose}\n"
                consultation_summary += f"   Key Insights: {consultation.key_insights}\n"
        
        synthesis_prompt = f"""
Based on your approach and the consultation results, provide the final response to the user.

Original Request: {user_input}

Your Planned Approach: {approach}

{consultation_summary}

Now provide the complete, final response that addresses the user's request. Incorporate insights from consultations where relevant, but ensure the response is cohesive and directly addresses what the user asked for.

Final Response:
"""
        
        final_output = await primary_agent.chat(synthesis_prompt)
        self.primary_tokens += self._estimate_tokens(synthesis_prompt + final_output)
        self.total_tokens += self.primary_tokens
        
        return final_output

    def _get_agent(self, model_name: str):
        """Get the appropriate agent for a model"""
        model_map = {
            "claude-4": "anthropic",
            "gpt-4o": "openai", 
            "gemini-2.0-pro": "gemini",
            "gemini-2.0-flash": "gemini"
        }
        
        agent_type = model_map.get(model_name, "anthropic")
        return self.agents[agent_type]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text.split()) * 1.3  # Approximate tokens

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
                model="gemini-2.0-flash",  # Fast model for search
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
