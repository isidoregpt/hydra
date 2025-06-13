"""
Updated ConsultationEngine with Gemini 2.5 model selection and thinking modes
"""
import asyncio
import json
from typing import Dict, Any, Optional, List

class ConsultationEngine:
    """Enhanced consultation engine with smart model selection"""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.web_search = WebSearchTool()
    
    async def execute_consultation(self, request: 'ConsultationRequest') -> 'ConsultationResult':
        """Execute a consultation with intelligent model selection and thinking modes"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Get the appropriate agent
            agent = self._get_agent_for_model(request.model)
            
            # Switch Gemini model if needed
            if hasattr(agent, 'switch_model') and 'gemini' in request.model:
                model_variant = self._get_gemini_variant(request.model)
                agent.switch_model(model_variant)
            
            # Build consultation prompt based on tool type
            prompt = await self._build_consultation_prompt(request)
            
            # Get thinking mode from context
            thinking_mode = request.context.get('thinking_mode')
            
            # Execute the consultation with thinking mode if supported
            if hasattr(agent, 'chat') and thinking_mode:
                response = await agent.chat(prompt, thinking_mode=thinking_mode)
            else:
                response = await agent.chat(prompt)
            
            # Extract key insights
            insights = self._extract_insights(response, request.tool)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return ConsultationResult(
                model=request.model,
                tool=request.tool,
                purpose=request.purpose,
                output=response,
                key_insights=insights,
                execution_time=execution_time,
                tokens_used=self._estimate_tokens(prompt + response)
            )
            
        except Exception as e:
            return ConsultationResult(
                model=request.model,
                tool=request.tool,
                purpose=request.purpose,
                output=f"Consultation failed: {str(e)}",
                key_insights="Error during consultation",
                execution_time=asyncio.get_event_loop().time() - start_time,
                tokens_used=0
            )
    
    def _get_gemini_variant(self, model_name: str) -> str:
        """Map model names to Gemini variants"""
        if "2.5-pro" in model_name or "pro" in model_name:
            return "2.5-pro"
        elif "2.5-flash" in model_name or "flash" in model_name:
            return "2.5-flash"
        else:
            return "2.5-flash"  # Default
    
    async def _build_consultation_prompt(self, request: 'ConsultationRequest') -> str:
        """Build specialized prompts based on consultation tool"""
        
        complexity = request.context.get('complexity', 'moderate')
        keywords = request.context.get('keywords', [])
        
        base_context = f"""
You are a specialist consultant providing expert analysis.

Consultation Purpose: {request.purpose}
Tool: {request.tool}
Task Complexity: {complexity}
Context: {request.context.get('user_input', '')}

Primary Agent's Approach: {request.context.get('approach', '')}
"""
        
        if request.tool == "analyze":
            return f"""{base_context}

Provide deep technical analysis focusing on:
- Architecture and design patterns
- Code quality and maintainability  
- Performance characteristics
- Potential improvements and optimizations

Use your expertise to provide insights that complement the primary analysis.

Your specialized analysis:"""
        
        elif request.tool == "debug":
            return f"""{base_context}

Provide systematic debugging analysis:
1. **Hypothesis Ranking**: Most likely causes first, with confidence levels
2. **Diagnostic Steps**: Specific steps to validate each hypothesis
3. **Root Cause Analysis**: Methodology to identify the underlying issue
4. **Prevention Strategies**: How to prevent similar issues

Your debugging analysis:"""
        
        elif request.tool == "codereview":
            return f"""{base_context}

Provide professional code review focusing on:
- 🔴 **Critical Issues**: Security vulnerabilities, data corruption risks
- 🟠 **High Priority**: Performance bottlenecks, logic errors
- 🟡 **Medium Priority**: Code quality, maintainability issues
- 🟢 **Low Priority**: Style and documentation improvements

Rate each issue by severity and provide specific remediation steps.

Your code review:"""
        
        elif request.tool == "thinkdeep":
            return f"""{base_context}

Provide extended reasoning and deep analysis:
- **Challenge Assumptions**: Question underlying assumptions in the approach
- **Explore Alternatives**: Consider different strategies and approaches
- **Identify Edge Cases**: Find potential failure modes and corner cases
- **Strategic Considerations**: Long-term implications and trade-offs
- **Risk Assessment**: Evaluate potential risks and mitigation strategies

Your deep analysis:"""
        
        elif request.tool == "websearch":
            # Special handling for web search tool
            search_query = request.context.get('search_query', request.purpose)
            async with self.web_search as search_tool:
                search_result = await search_tool.search(search_query)
                
                return f"""{base_context}

Web Search Results for: {search_query}
{json.dumps(search_result, indent=2)}

Based on these search results, provide analysis and recommendations:
- Summarize the key findings
- Identify actionable information
- Suggest next steps or additional searches if needed

Your search analysis:"""
        
        else:
            return f"""{base_context}

Provide expert consultation in your area of specialization.
Consider the task complexity ({complexity}) and focus on delivering
insights that add value to the primary analysis.

Your analysis:"""
    
    def _get_agent_for_model(self, model_name: str):
        """Map model names to appropriate agents"""
        if "gpt" in model_name.lower() or "4o" in model_name:
            return self.agents["openai"]
        elif "gemini" in model_name.lower() or "flash" in model_name or "pro" in model_name:
            return self.agents["gemini"]
        elif "claude" in model_name.lower():
            return self.agents["anthropic"]
        else:
            # Default to anthropic
            return self.agents["anthropic"]
    
    def _extract_insights(self, response: str, tool: str) -> str:
        """Extract key insights based on tool type"""
        lines = response.split('\n')
        insights = []
        
        # Tool-specific insight extraction with enhanced patterns
        if tool == "codereview":
            # Look for severity indicators and security issues
            for line in lines:
                line_lower = line.lower()
                if any(indicator in line_lower for indicator in ['🔴', '🟠', 'critical', 'security', 'vulnerability', 'high priority']):
                    insights.append(line.strip())
                elif any(pattern in line_lower for pattern in ['issue:', 'problem:', 'concern:', 'risk:']):
                    insights.append(line.strip())
                    
        elif tool == "debug":
            # Look for hypotheses, root causes, and diagnostic steps
            for line in lines:
                line_lower = line.lower()
                if any(indicator in line_lower for indicator in ['hypothesis', 'root cause', 'likely cause', 'diagnostic']):
                    insights.append(line.strip())
                elif any(pattern in line_lower for pattern in ['1.', '2.', '3.']) and any(word in line_lower for word in ['cause', 'issue', 'problem']):
                    insights.append(line.strip())
                    
        elif tool == "thinkdeep":
            # Look for strategic insights and assumptions
            for line in lines:
                line_lower = line.lower()
                if any(indicator in line_lower for indicator in ['assumption', 'alternative', 'edge case', 'risk', 'strategy']):
                    insights.append(line.strip())
                elif line.startswith('**') and line.endswith('**'):  # Bold headings
                    insights.append(line.strip())
                    
        elif tool == "analyze":
            # Look for architectural and performance insights
            for line in lines:
                line_lower = line.lower()
                if any(indicator in line_lower for indicator in ['architecture', 'pattern', 'performance', 'optimization', 'improvement']):
                    insights.append(line.strip())
                    
        else:
            # General insight extraction
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in ['key', 'important', 'critical', 'recommend', 'suggest', 'conclusion']):
                    insights.append(line.strip())
                elif line.startswith('**') or line.startswith('##'):  # Headers
                    insights.append(line.strip())
        
        # Return top insights, ensuring we have meaningful content
        filtered_insights = [insight for insight in insights if len(insight) > 10]
        return '\n'.join(filtered_insights[:4])  # Top 4 insights
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text.split()) * 1.3


# Keep the rest of the existing classes (WebSearchTool, ToolRegistry, etc.)
# from the original tools_system_complete.py file...

class WebSearchTool:
    """Enhanced web search with multiple APIs and fallbacks"""
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        import aiohttp
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search(self, query: str, search_type: str = "general") -> Dict:
        """
        Perform web search with multiple fallback strategies
        """
        if not self.session:
            import aiohttp
            self.session = aiohttp.ClientSession()
        
        # Try different search strategies based on query type
        try:
            if self._is_date_time_query(query):
                return await self._handle_date_time_query(query)
            elif self._is_technical_query(query):
                return await self._handle_technical_query(query)
            elif self._is_news_query(query):
                return await self._handle_news_query(query)
            else:
                return await self._handle_general_query(query)
                
        except Exception as e:
            return {
                "type": "search_failed",
                "query": query,
                "error": str(e),
                "suggestions": self._generate_search_suggestions(query)
            }
    
    async def _handle_date_time_query(self, query: str) -> Dict:
        """Handle date/time queries with real-time APIs"""
        try:
            # Try WorldTimeAPI first
            url = "http://worldtimeapi.org/api/timezone/Etc/UTC"
            async with self.session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    datetime_str = data.get('datetime', '')
                    
                    if datetime_str:
                        from datetime import datetime
                        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                        
                        return {
                            "type": "current_info",
                            "query": query,
                            "answer": f"Today is {dt.strftime('%A, %B %d, %Y')}. Current UTC time is {dt.strftime('%H:%M:%S')}.",
                            "source": "WorldTimeAPI",
                            "local_note": "This is UTC time. Your local time may differ based on timezone."
                        }
        except:
            pass
        
        # Fallback for date/time queries
        return {
            "type": "search_suggestion",
            "query": query,
            "answer": "I cannot access real-time date/time information.",
            "suggestions": [
                "Check your device's system clock",
                "Search 'current date and time' on Google",
                "Visit timeanddate.com for accurate time information"
            ]
        }
    
    def _is_date_time_query(self, query: str) -> bool:
        """Check if query is asking for current date/time"""
        query_lower = query.lower()
        indicators = [
            'today', 'date', 'time', 'current date', 'what day',
            'today\'s date', 'what time', 'current time', 'now'
        ]
        return any(indicator in query_lower for indicator in indicators)
    
    def _is_technical_query(self, query: str) -> bool:
        """Check if query is technical/programming related"""
        query_lower = query.lower()
        tech_indicators = [
            'python', 'javascript', 'react', 'api', 'code', 'programming',
            'framework', 'library', 'database', 'sql', 'html', 'css',
            'docker', 'kubernetes', 'aws', 'github', 'git'
        ]
        return any(indicator in query_lower for indicator in tech_indicators)
    
    def _is_news_query(self, query: str) -> bool:
        """Check if query is asking for news/current events"""
        query_lower = query.lower()
        news_indicators = [
            'news', 'latest', 'recent', 'current events', 'breaking',
            'update', 'today\'s news', 'what happened'
        ]
        return any(indicator in query_lower for indicator in news_indicators)
    
    async def _handle_technical_query(self, query: str) -> Dict:
        """Handle technical documentation queries"""
        tech_keywords = self._extract_tech_keywords(query)
        
        return {
            "type": "search_suggestion", 
            "query": query,
            "answer": "For technical information, I recommend searching these resources:",
            "suggestions": [
                f"Official documentation for {', '.join(tech_keywords[:3])}",
                f"Stack Overflow: {query}",
                f"GitHub issues: {query}",
                f"Reddit r/programming: {query}"
            ],
            "search_urls": [
                f"https://www.google.com/search?q={query}+documentation",
                f"https://stackoverflow.com/search?q={query}",
                f"https://github.com/search?q={query}&type=issues"
            ]
        }
    
    async def _handle_news_query(self, query: str) -> Dict:
        """Handle news and current events queries"""
        return {
            "type": "search_suggestion",
            "query": query, 
            "answer": "For current news and events, check these sources:",
            "suggestions": [
                f"Google News: {query}",
                f"Reuters: {query}",
                f"Associated Press: {query}",
                "Your preferred news website"
            ],
            "search_urls": [
                f"https://news.google.com/search?q={query}",
                f"https://www.reuters.com/search/news?blob={query}",
                f"https://apnews.com/search?q={query}"
            ]
        }
    
    async def _handle_general_query(self, query: str) -> Dict:
        """Handle general web search queries"""
        return {
            "type": "search_suggestion",
            "query": query,
            "answer": f"To find information about '{query}', I recommend:",
            "suggestions": [
                f"Google search: {query}",
                f"Wikipedia: {query}",
                f"DuckDuckGo: {query}",
                "Academic sources if research-related"
            ],
            "search_urls": [
                f"https://www.google.com/search?q={query}",
                f"https://en.wikipedia.org/wiki/Special:Search/{query}",
                f"https://duckduckgo.com/?q={query}"
            ]
        }
    
    def _extract_tech_keywords(self, query: str) -> List[str]:
        """Extract technology keywords from query"""
        common_techs = [
            'python', 'javascript', 'react', 'node', 'vue', 'angular',
            'django', 'flask', 'fastapi', 'express', 'spring', 'java',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git'
        ]
        
        query_lower = query.lower()
        found_techs = [tech for tech in common_techs if tech in query_lower]
        return found_techs or ['development']
    
    def _generate_search_suggestions(self, query: str) -> List[str]:
        """Generate helpful search suggestions when web search fails"""
        return [
            f"Google: {query}",
            f"DuckDuckGo: {query}",
            f"Wikipedia: {query}",
            "Try rephrasing your question",
            "Check if you're connected to the internet"
        ]


# Data models for consultation system
from dataclasses import dataclass
from typing import List

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

# Tool registry and other classes remain the same...
from enum import Enum

class ToolType(Enum):
    ANALYSIS = "analysis"
    REVIEW = "review"
    DEBUG = "debug"
    CREATIVE = "creative"
    SEARCH = "search"
    VALIDATION = "validation"

@dataclass
class Tool:
    name: str
    description: str
    type: ToolType
    best_models: List[str]
    thinking_modes: List[str]
    use_cases: List[str]

class ToolRegistry:
    """Registry of available consultation tools"""
    
    def __init__(self):
        self.tools = {
            "analyze": Tool(
                name="analyze",
                description="Deep analysis of code, architecture, and systems",
                type=ToolType.ANALYSIS,
                best_models=["gemini-2.5-pro", "gemini-2.5-flash", "gpt-4o"],
                thinking_modes=["medium", "high", "max"],
                use_cases=[
                    "Understanding complex codebases",
                    "Architecture analysis",
                    "System design evaluation", 
                    "Pattern identification"
                ]
            ),
            
            "codereview": Tool(
                name="codereview", 
                description="Professional code review with security and quality focus",
                type=ToolType.REVIEW,
                best_models=["gemini-2.5-pro", "gemini-2.5-flash", "gpt-4o"],
                thinking_modes=["medium", "high"],
                use_cases=[
                    "Security vulnerability assessment",
                    "Code quality evaluation",
                    "Best practices compliance",
                    "Performance optimization"
                ]
            ),
            
            "debug": Tool(
                name="debug",
                description="Root cause analysis and systematic debugging",
                type=ToolType.DEBUG, 
                best_models=["gpt-4o", "gemini-2.5-pro", "gemini-2.5-flash"],
                thinking_modes=["medium", "high"],
                use_cases=[
                    "Error diagnosis and resolution",
                    "Performance bottleneck identification",
                    "Logic error analysis",
                    "System failure investigation"
                ]
            ),
            
            "thinkdeep": Tool(
                name="thinkdeep",
                description="Extended reasoning for complex problems",
                type=ToolType.ANALYSIS,
                best_models=["gemini-2.5-pro"],
                thinking_modes=["high", "max"],
                use_cases=[
                    "Complex architectural decisions",
                    "Strategic technical planning",
                    "Multi-faceted problem solving",
                    "Risk analysis"
                ]
            ),
            
            "websearch": Tool(
                name="websearch",
                description="Web search for current information",
                type=ToolType.SEARCH,
                best_models=["gemini-2.5-flash", "gpt-4o"],
                thinking_modes=["low", "medium"],
                use_cases=[
                    "Current documentation lookup",
                    "Latest news and events",
                    "Technical problem solutions",
                    "Real-time data queries"
                ]
            )
        }
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get tool configuration by name"""
        return self.tools.get(tool_name)
    
    def suggest_tools_for_task(self, task_category: str, keywords: List[str]) -> List[str]:
        """Suggest appropriate tools based on task characteristics"""
        suggestions = []
        
        # Category-based suggestions
        category_tools = {
            "creative": [],  # Handle directly
            "technical": ["analyze", "codereview", "debug"],
            "analytical": ["analyze", "thinkdeep"],
            "review": ["codereview"],
            "planning": ["thinkdeep", "analyze"],
            "troubleshooting": ["debug", "analyze"],
            "current_info": ["websearch"]
        }
        
        suggestions.extend(category_tools.get(task_category, ["analyze"]))
        
        # Keyword-based suggestions
        if any(keyword in keywords for keyword in ['security', 'vulnerability']):
            if 'codereview' not in suggestions:
                suggestions.append('codereview')
        
        if any(keyword in keywords for keyword in ['performance', 'optimization']):
            if 'analyze' not in suggestions:
                suggestions.append('analyze')
        
        if any(keyword in keywords for keyword in ['current', 'latest', 'news', 'today']):
            if 'websearch' not in suggestions:
                suggestions.append('websearch')
        
        return suggestions[:3]  # Limit to top 3
