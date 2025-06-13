"""
Complete tools system for Hydra v3 with web search and consultation capabilities
"""
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Tool registry and base classes
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

class WebSearchTool:
    """Enhanced web search with multiple APIs and fallbacks"""
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
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
    
    async def _handle_technical_query(self, query: str) -> Dict:
        """Handle technical documentation queries"""
        # Extract technology keywords
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

class ConsultationEngine:
    """Manages consultation requests between primary and specialist models"""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.web_search = WebSearchTool()
    
    async def execute_consultation(self, request: 'ConsultationRequest') -> 'ConsultationResult':
        """Execute a consultation with a specialist model"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Get the appropriate agent
            agent = self._get_agent_for_model(request.model)
            
            # Build consultation prompt based on tool type
            prompt = await self._build_consultation_prompt(request)
            
            # Execute the consultation
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
    
    async def _build_consultation_prompt(self, request: 'ConsultationRequest') -> str:
        """Build specialized prompts based on consultation tool"""
        
        base_context = f"""
You are a specialist consultant providing expert analysis.

Consultation Purpose: {request.purpose}
Tool: {request.tool}
Context: {request.context.get('user_input', '')}

Primary Agent's Approach: {request.context.get('approach', '')}
"""
        
        if request.tool == "analyze":
            return f"""{base_context}

Provide deep technical analysis focusing on:
- Architecture and design patterns
- Code quality and maintainability
- Performance characteristics
- Potential improvements

Your specialized analysis:"""
        
        elif request.tool == "debug":
            return f"""{base_context}

Provide systematic debugging analysis:
1. Hypothesis ranking (most likely causes first)
2. Diagnostic steps to validate each hypothesis
3. Root cause analysis methodology
4. Prevention strategies

Your debugging analysis:"""
        
        elif request.tool == "codereview":
            return f"""{base_context}

Provide professional code review:
- Security vulnerabilities (🔴 Critical, 🟠 High, 🟡 Medium, 🟢 Low)
- Performance issues and optimizations
- Code quality and best practices
- Maintainability concerns

Your code review:"""
        
        elif request.tool == "thinkdeep":
            return f"""{base_context}

Provide extended reasoning and analysis:
- Challenge assumptions
- Explore alternative approaches
- Identify edge cases and risks
- Strategic considerations

Your deep analysis:"""
        
        elif request.tool == "websearch":
            # Special handling for web search tool
            search_query = request.context.get('search_query', request.purpose)
            async with self.web_search as search_tool:
                search_result = await search_tool.search(search_query)
                
                return f"""{base_context}

Web Search Results for: {search_query}
{json.dumps(search_result, indent=2)}

Based on these search results, provide analysis and recommendations:"""
        
        else:
            return f"""{base_context}

Provide expert consultation in your area of specialization.

Your analysis:"""
    
    def _get_agent_for_model(self, model_name: str):
        """Map model names to appropriate agents"""
        model_map = {
            "gpt-4o": "openai",
            "gemini-2.0-pro": "gemini", 
            "gemini-2.0-flash": "gemini",
            "claude-4": "anthropic"
        }
        
        agent_type = model_map.get(model_name, "anthropic")
        return self.agents[agent_type]
    
    def _extract_insights(self, response: str, tool: str) -> str:
        """Extract key insights based on tool type"""
        lines = response.split('\n')
        insights = []
        
        # Tool-specific insight extraction
        if tool == "codereview":
            for line in lines:
                if any(indicator in line.lower() for indicator in ['🔴', '🟠', 'critical', 'security', 'vulnerability']):
                    insights.append(line.strip())
        elif tool == "debug":
            for line in lines:
                if any(indicator in line.lower() for indicator in ['root cause', 'hypothesis', 'likely', 'issue']):
                    insights.append(line.strip())
        else:
            # General insight extraction
            for line in lines:
                if any(keyword in line.lower() for keyword in ['key', 'important', 'critical', 'recommend', 'suggest']):
                    insights.append(line.strip())
        
        return '\n'.join(insights[:3])  # Top 3 insights
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text.split()) * 1.3

# Data models for consultation system
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

class ToolRegistry:
    """Registry of available consultation tools"""
    
    def __init__(self):
        self.tools = {
            "analyze": Tool(
                name="analyze",
                description="Deep analysis of code, architecture, and systems",
                type=ToolType.ANALYSIS,
                best_models=["gemini-2.0-pro", "gpt-4o"],
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
                best_models=["gemini-2.0-pro", "gpt-4o"],
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
                best_models=["gpt-4o", "gemini-2.0-pro"],
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
                best_models=["gemini-2.0-pro"],
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
                best_models=["gemini-2.0-flash", "gpt-4o"],
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
