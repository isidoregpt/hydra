import aiohttp
import asyncio
from typing import Dict, List, Optional
import json
import urllib.parse

class WebSearchTool:
    """
    Web search functionality for getting current information.
    Uses DuckDuckGo Instant Answer API for simple queries.
    """
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search(self, query: str, max_results: int = 5) -> Dict:
        """
        Perform a web search and return results.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            Dict with search results and metadata
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            # Try DuckDuckGo Instant Answer API first for direct answers
            instant_result = await self._get_instant_answer(query)
            if instant_result:
                return {
                    "type": "instant_answer",
                    "query": query,
                    "answer": instant_result,
                    "source": "DuckDuckGo Instant Answer"
                }
            
            # For date/time queries, use a simple approach
            if self._is_date_time_query(query):
                return await self._handle_date_time_query(query)
            
            # For other queries, return search suggestions
            return {
                "type": "search_needed",
                "query": query,
                "suggestion": f"I recommend searching for: '{query}' to get current information.",
                "search_engines": [
                    f"https://duckduckgo.com/?q={urllib.parse.quote(query)}",
                    f"https://www.google.com/search?q={urllib.parse.quote(query)}"
                ]
            }
            
        except Exception as e:
            return {
                "type": "error",
                "query": query,
                "error": f"Search failed: {str(e)}",
                "fallback": f"Please search manually for: {query}"
            }
    
    async def _get_instant_answer(self, query: str) -> Optional[str]:
        """Get instant answer from DuckDuckGo API."""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check for direct answer
                    if data.get('Answer'):
                        return data['Answer']
                    
                    # Check for abstract
                    if data.get('Abstract'):
                        return data['Abstract']
                    
                    # Check for definition
                    if data.get('Definition'):
                        return data['Definition']
                        
        except Exception:
            pass
        
        return None
    
    def _is_date_time_query(self, query: str) -> bool:
        """Check if query is asking for current date/time."""
        query_lower = query.lower()
        date_time_indicators = [
            'today', 'date', 'time', 'current date', 'what day',
            'today\'s date', 'what time', 'current time'
        ]
        return any(indicator in query_lower for indicator in date_time_indicators)
    
    async def _handle_date_time_query(self, query: str) -> Dict:
        """Handle date/time queries with current information."""
        try:
            # Use a time API service
            url = "http://worldtimeapi.org/api/timezone/Etc/UTC"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    datetime_str = data.get('datetime', '')
                    
                    if datetime_str:
                        # Parse the datetime
                        from datetime import datetime
                        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                        
                        formatted_date = dt.strftime("%A, %B %d, %Y")
                        formatted_time = dt.strftime("%H:%M:%S UTC")
                        
                        return {
                            "type": "current_info",
                            "query": query,
                            "answer": f"Today is {formatted_date}. Current UTC time is {formatted_time}.",
                            "source": "WorldTimeAPI"
                        }
        except Exception:
            pass
        
        # Fallback response
        return {
            "type": "search_needed",
            "query": query,
            "suggestion": "I cannot access real-time date/time information. Please check your device's clock or search for 'current date and time'.",
            "fallback": "Check your device's system clock or calendar app."
        }

    async def suggest_searches(self, context: str, analysis: str) -> List[str]:
        """
        Suggest web searches that would enhance the analysis.
        This mimics the Zen MCP approach of recommending searches to Claude.
        """
        suggestions = []
        
        context_lower = context.lower()
        analysis_lower = analysis.lower()
        
        # Technical documentation searches
        if any(tech in context_lower for tech in ['api', 'framework', 'library', 'package']):
            suggestions.append(f"latest documentation for {self._extract_tech_stack(context)}")
        
        # Error message searches
        if any(error in context_lower for error in ['error', 'exception', 'bug', 'issue']):
            error_msg = self._extract_error_message(context)
            if error_msg:
                suggestions.append(f'"{error_msg}" solution 2024')
        
        # Security best practices
        if any(sec in analysis_lower for sec in ['security', 'vulnerability', 'authentication']):
            suggestions.append("current security best practices " + self._extract_tech_stack(context))
        
        # Performance optimization
        if any(perf in analysis_lower for perf in ['performance', 'optimization', 'slow']):
            suggestions.append(f"performance optimization {self._extract_tech_stack(context)} 2024")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def _extract_tech_stack(self, text: str) -> str:
        """Extract technology stack from text."""
        common_techs = [
            'python', 'javascript', 'react', 'node', 'fastapi', 'django',
            'flask', 'express', 'vue', 'angular', 'typescript', 'java',
            'spring', 'docker', 'kubernetes', 'aws', 'postgresql', 'mongodb'
        ]
        
        text_lower = text.lower()
        found_techs = [tech for tech in common_techs if tech in text_lower]
        
        return ' '.join(found_techs[:2]) if found_techs else 'development'
    
    def _extract_error_message(self, text: str) -> Optional[str]:
        """Extract error message from text."""
        # Look for common error patterns
        error_patterns = [
            r'Error: (.+)',
            r'Exception: (.+)',
            r'TypeError: (.+)',
            r'ValueError: (.+)',
            r'AttributeError: (.+)'
        ]
        
        import re
        for pattern in error_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)[:50]  # First 50 chars
        
        return None
