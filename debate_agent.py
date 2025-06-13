class DebateAgent:
    def __init__(self, arbitration_agent):
        self.arbitration_agent = arbitration_agent

    async def arbitrate(self, task, responses, critiques):
        """
        Arbitrate between multiple responses using critiques to determine the best synthesis.
        
        Args:
            task: The original task/question
            responses: Dict with keys OpenAI, Claude, Gemini containing their responses
            critiques: Dict with keys OpenAI, Claude, Gemini containing their critiques
            
        Returns:
            String containing the arbitrated final response
        """
        arbitration_prompt = f"""
You are an expert arbiter tasked with synthesizing the best final answer from multiple AI responses and their peer critiques.

Original Task: {task}

Candidate Responses:
- OpenAI: {responses.get('OpenAI', 'No response')}
- Claude: {responses.get('Claude', 'No response')}  
- Gemini: {responses.get('Gemini', 'No response')}

Peer Critiques:
- OpenAI's critique: {critiques.get('OpenAI', 'No critique')}
- Claude's critique: {critiques.get('Claude', 'No critique')}
- Gemini's critique: {critiques.get('Gemini', 'No critique')}

Instructions:
1. Analyze each response for accuracy, completeness, and quality
2. Consider the validity of each critique
3. Identify the strongest elements from each response
4. Synthesize a final answer that incorporates the best aspects while addressing identified weaknesses
5. Provide a clear, actionable final response

Final Arbitrated Response:
"""
        try:
            result = await self.arbitration_agent.chat(arbitration_prompt)
            return result.strip()
        except Exception as e:
            # Fallback if arbitration fails
            return f"Arbitration failed: {str(e)}. Defaulting to first available response: {responses.get('OpenAI', responses.get('Claude', responses.get('Gemini', 'No responses available')))}"
