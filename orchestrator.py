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
    Claude Opus 4 orchestrated multi-round collaborative system.
    Uses the most capable Claude model for primary orchestration.
    """
    
    def __init__(
        self,
        agents: Dict[str, Any],
        primary_model: str = "claude-opus-4",
        available_consultants: List[str] = None,
        auto_consultation: bool = True,
        max_consultations: int = 4,
        thinking_depth: str = "high",
        enable_web_search: bool = True,
        memory = None
    ):
        self.agents = agents
        self.primary_model = primary_model
        # Enhanced default consultants including Claude Sonnet 4
        self.available_consultants = available_consultants or ["claude-sonnet-4", "gpt-4o", "gemini-2.5-pro"]
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
        Multi-round collaborative execution with Claude Opus 4 orchestration
        """
        self.start_time = time.time()
        self.consultation_count = 0
        self.primary_tokens = 0
        self.total_tokens = 0
        
        if progress_callback:
            progress_callback("Claude Opus 4 analyzing task complexity...", 0.05)
        
        # Check if this is a simple time query
        if self._is_simple_time_query(user_input):
            return await self._handle_simple_query(user_input, session_id, progress_callback)
        
        # Multi-round collaborative process with Claude Opus 4
        all_consultations = []
        
        # ROUND 1: Independent analysis by all available models
        if progress_callback:
            progress_callback("Round 1: Independent analysis by specialist models...", 0.1)
        
        round1_consultations = await self._round1_independent_analysis(user_input, progress_callback)
        all_consultations.extend(round1_consultations)
        
        # ROUND 2: Cross-review and refinement
        if progress_callback:
            progress_callback("Round 2: Cross-review and constructive criticism...", 0.4)
        
        round2_consultations = await self._round2_cross_review(user_input, round1_consultations, progress_callback)
        all_consultations.extend(round2_consultations)
        
        # ROUND 3: Consensus building with best available model
        if progress_callback:
            progress_callback("Round 3: Building expert consensus...", 0.7)
        
        round3_consultations = await self._round3_consensus(user_input, round1_consultations, round2_consultations, progress_callback)
        all_consultations.extend(round3_consultations)
        
        # FINAL: Claude Opus 4 synthesizes the collaborative result
        if progress_callback:
            progress_callback("Claude Opus 4 synthesizing collaborative insights...", 0.9)
        
        final_output = await self._synthesize_collaborative_result(user_input, all_consultations)
        
        if progress_callback:
            progress_callback("Complete!", 1.0)
        
        # Calculate final metrics
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
                for c in all_consultations
            ],
            "collaboration_rounds": {
                "round_1_independent": len(round1_consultations),
                "round_2_cross_review": len(round2_consultations),
                "round_3_consensus": len(round3_consultations)
            },
            "metrics": {
                "total_time": execution_time,
                "consultations_count": self.consultation_count,
                "primary_tokens": self.primary_tokens,
                "total_tokens": self.total_tokens,
                "task_complexity": "claude_opus_4_collaborative",
                "primary_model": self.primary_model
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id
        }
        
        if self.memory:
            self.memory.save_session(session_id, result)
        
        return result

    async def _round1_independent_analysis(self, user_input: str, progress_callback: Optional[Callable] = None) -> List[ConsultationResult]:
        """Round 1: Each specialist model provides independent analysis"""
        consultations = []
        
        # Get all available consultant models
        models_to_consult = self.available_consultants.copy()
        
        consultation_requests = []
        for i, model in enumerate(models_to_consult):
            # Customize approach based on each model's capabilities
            if "claude-sonnet-4" in model:
                consultation_requests.append(ConsultationRequest(
                    purpose=f"Independent analysis (Round 1) by Claude Sonnet 4",
                    model=model,
                    tool="analyze",
                    context={
                        "user_input": user_input,
                        "round": 1,
                        "instruction": "Provide independent analysis using Claude Sonnet 4's exceptional reasoning capabilities."
                    }
                ))
            elif "gemini-2.5-pro" in model:
                consultation_requests.append(ConsultationRequest(
                    purpose=f"Independent deep analysis (Round 1) by Gemini 2.5 Pro",
                    model=model,
                    tool="thinkdeep",
                    context={
                        "user_input": user_input,
                        "thinking_mode": self.thinking_depth,
                        "round": 1,
                        "instruction": "Provide deep independent analysis with maximum thinking depth."
                    }
                ))
            elif "gemini-2.5-flash" in model:
                consultation_requests.append(ConsultationRequest(
                    purpose=f"Independent efficient analysis (Round 1) by Gemini 2.5 Flash",
                    model=model,
                    tool="analyze",
                    context={
                        "user_input": user_input,
                        "thinking_mode": "medium",
                        "round": 1,
                        "instruction": "Provide efficient independent analysis with adaptive thinking."
                    }
                ))
            else:  # GPT-4o
                consultation_requests.append(ConsultationRequest(
                    purpose=f"Independent analysis (Round 1) by GPT-4o",
                    model=model,
                    tool="analyze",
                    context={
                        "user_input": user_input,
                        "round": 1,
                        "instruction": "Provide independent analysis with focus on practical insights and recommendations."
                    }
                ))
        
        # Execute round 1 consultations
        for i, req in enumerate(consultation_requests):
            if progress_callback:
                progress_percent = 0.1 + (0.25 * (i / len(consultation_requests)))
                progress_callback(f"Round 1: {req.model} analyzing independently...", progress_percent)
            
            consultation_result = await self.consultation_engine.execute_consultation(req)
            consultations.append(consultation_result)
            self.consultation_count += 1
        
        return consultations

    async def _round2_cross_review(self, user_input: str, round1_consultations: List[ConsultationResult], progress_callback: Optional[Callable] = None) -> List[ConsultationResult]:
        """Round 2: Models review each other's analyses with constructive criticism"""
        consultations = []
        
        # Create comprehensive summary of round 1 results
        round1_summary = self._create_detailed_round_summary(round1_consultations, "Round 1 Independent Analyses")
        
        # Each model provides cross-review
        models_to_consult = self.available_consultants.copy()
        
        consultation_requests = []
        for model in models_to_consult:
            consultation_requests.append(ConsultationRequest(
                purpose=f"Cross-review and refinement (Round 2) by {model}",
                model=model,
                tool="analyze",
                context={
                    "user_input": user_input,
                    "round1_analyses": round1_summary,
                    "thinking_mode": "high" if "gemini" in model else None,
                    "round": 2,
                    "instruction": f"""Review and critique the other models' Round 1 analyses.
                    
                    Your cross-review tasks:
                    1. **Identify Strengths**: What insights were particularly valuable?
                    2. **Spot Weaknesses**: Where were analyses incomplete or inaccurate?
                    3. **Find Contradictions**: Do models disagree? Which perspective is stronger?
                    4. **Suggest Improvements**: How could each analysis be enhanced?
                    5. **Synthesize Best Ideas**: Combine the strongest insights from all models
                    6. **Add New Perspectives**: What did others miss that you can contribute?
                    
                    Be constructive but honest in your criticism. Focus on improving the collective analysis."""
                }
            ))
        
        # Execute round 2 consultations
        for i, req in enumerate(consultation_requests):
            if progress_callback:
                progress_percent = 0.4 + (0.25 * (i / len(consultation_requests)))
                progress_callback(f"Round 2: {req.model} cross-reviewing analyses...", progress_percent)
            
            consultation_result = await self.consultation_engine.execute_consultation(req)
            consultations.append(consultation_result)
            self.consultation_count += 1
        
        return consultations

    async def _round3_consensus(self, user_input: str, round1_consultations: List[ConsultationResult], round2_consultations: List[ConsultationResult], progress_callback: Optional[Callable] = None) -> List[ConsultationResult]:
        """Round 3: Models work toward consensus"""
        consultations = []
        
        # Create comprehensive summary of all previous rounds
        all_previous = self._create_comprehensive_summary(round1_consultations, round2_consultations)
        
        # Select the best model for consensus building (prefer Claude Sonnet 4 or Gemini 2.5 Pro)
        consensus_model = "claude-sonnet-4" if "claude-sonnet-4" in self.available_consultants else \
                         "gemini-2.5-pro" if "gemini-2.5-pro" in self.available_consultants else \
                         self.available_consultants[0]
        
        consultation_request = ConsultationRequest(
            purpose=f"Consensus building (Round 3) by {consensus_model}",
            model=consensus_model,
            tool="thinkdeep" if "gemini" in consensus_model else "analyze",
            context={
                "user_input": user_input,
                "all_previous_rounds": all_previous,
                "thinking_mode": "max" if "gemini" in consensus_model else None,
                "round": 3,
                "instruction": """Based on all previous analyses and cross-reviews, build a consensus.

                Your consensus-building tasks:
                1. **Identify Strong Consensus**: What do all models agree on?
                2. **Resolve Contradictions**: Where models disagree, weigh evidence and expertise
                3. **Integrate Best Insights**: Combine the strongest points from all perspectives
                4. **Fill Gaps**: Address any important aspects that were missed
                5. **Create Unified Recommendations**: Provide coherent, actionable guidance
                6. **Note Uncertainties**: Highlight any remaining areas of disagreement
                
                Build a consensus that incorporates the collective wisdom of all models."""
            }
        )
        
        if progress_callback:
            progress_callback(f"Round 3: {consensus_model} building consensus...", 0.75)
        
        consultation_result = await self.consultation_engine.execute_consultation(consultation_request)
        consultations.append(consultation_result)
        self.consultation_count += 1
        
        return consultations

    async def _synthesize_collaborative_result(self, user_input: str, all_consultations: List[ConsultationResult]) -> str:
        """Claude Opus 4 synthesizes the final collaborative result"""
        
        # Organize consultations by round
        round1 = [c for c in all_consultations if "Round 1" in c.purpose]
        round2 = [c for c in all_consultations if "Round 2" in c.purpose]
        round3 = [c for c in all_consultations if "Round 3" in c.purpose]
        
        synthesis_prompt = f"""
You are Claude Opus 4, synthesizing the results of a comprehensive multi-round collaborative analysis.

USER REQUEST: {user_input}

=== MULTI-ROUND COLLABORATIVE ANALYSIS ===

ROUND 1 - INDEPENDENT ANALYSES:
{self._format_consultations_for_synthesis(round1)}

ROUND 2 - CROSS-REVIEWS AND CRITIQUES:
{self._format_consultations_for_synthesis(round2)}

ROUND 3 - CONSENSUS BUILDING:
{self._format_consultations_for_synthesis(round3)}

=== YOUR SYNTHESIS TASK ===

As Claude Opus 4, provide the definitive collaborative response that:

1. **Incorporates Collective Intelligence**: Synthesize insights from all models across all rounds
2. **Highlights Expert Consensus**: Emphasize areas where multiple models strongly agreed
3. **Resolves Remaining Conflicts**: Use your superior reasoning to address any contradictions
4. **Delivers Exceptional Value**: Provide clear, actionable, and comprehensive guidance
5. **Acknowledges the Collaborative Process**: Note how multi-round analysis enhanced the result
6. **Demonstrates Superior Reasoning**: Show why this collaborative approach produces better outcomes

Your final collaborative synthesis:
"""
        
        primary_agent = self._get_agent(self.primary_model)
        final_response = await primary_agent.chat(synthesis_prompt, max_tokens=4096)
        
        self.primary_tokens = self._estimate_tokens(synthesis_prompt + final_response)
        
        # Add all consultation tokens
        for consultation in all_consultations:
            self.total_tokens += consultation.tokens_used
        self.total_tokens += self.primary_tokens
        
        return final_response

    def _create_detailed_round_summary(self, consultations: List[ConsultationResult], title: str) -> str:
        """Create detailed summary of consultation results"""
        summary = f"\n=== {title} ===\n"
        for consultation in consultations:
            summary += f"\n**{consultation.model}**:\n"
            summary += f"Key Insights: {consultation.key_insights}\n"
            summary += f"Analysis: {consultation.output[:400]}{'...' if len(consultation.output) > 400 else ''}\n"
            summary += "---\n"
        return summary

    def _create_comprehensive_summary(self, round1: List[ConsultationResult], round2: List[ConsultationResult]) -> str:
        """Create comprehensive summary of all previous rounds"""
        summary = self._create_detailed_round_summary(round1, "Round 1: Independent Analyses")
        summary += self._create_detailed_round_summary(round2, "Round 2: Cross-Reviews and Refinements")
        return summary

    def _format_consultations_for_synthesis(self, consultations: List[ConsultationResult]) -> str:
        """Format consultations for the final synthesis"""
        if not consultations:
            return "No consultations in this round.\n"
        
        formatted = ""
        for consultation in consultations:
            formatted += f"\n**{consultation.model}** ({consultation.purpose}):\n"
            formatted += f"Key Insights: {consultation.key_insights}\n"
            formatted += f"Analysis: {consultation.output[:300]}{'...' if len(consultation.output) > 300 else ''}\n"
            formatted += "---\n"
        return formatted

    async def _handle_simple_query(self, user_input: str, session_id: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Handle simple time queries without multi-round consultation"""
        if progress_callback:
            progress_callback("Handling simple query...", 0.5)
        
        web_result = await self._handle_web_search(user_input)
        
        if progress_callback:
            progress_callback("Complete!", 1.0)
        
        return {
            "primary_output": web_result,
            "consultations": [],
            "collaboration_rounds": {"simple_query": True},
            "metrics": {
                "total_time": time.time() - self.start_time,
                "consultations_count": 0,
                "primary_tokens": 0,
                "total_tokens": 0,
                "task_complexity": "simple_query"
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id
        }

    def _is_simple_time_query(self, user_input: str) -> bool:
        """Check if this is ONLY asking for current time/date"""
        user_lower = user_input.lower().strip()
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
            "claude-opus-4": "anthropic",
            "claude-sonnet-4": "anthropic",
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
