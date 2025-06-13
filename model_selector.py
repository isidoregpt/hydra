"""
Smart model selection for Hydra v3 consultations - Updated for official Gemini 2.5 models
"""
from typing import Dict, List, Optional, Any
from enum import Enum

class ModelCapability(Enum):
    FAST_ANALYSIS = "fast_analysis"
    DEEP_THINKING = "deep_thinking" 
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    WEB_SEARCH = "web_search"

class ModelSelector:
    """
    Intelligent model selection for different consultation types
    """
    
    def __init__(self):
        # Updated model capabilities with official Gemini 2.5 models
        self.model_capabilities = {
            "gemini-2.5-flash": {
                "strengths": [
                    ModelCapability.FAST_ANALYSIS,
                    ModelCapability.CODE_REVIEW, 
                    ModelCapability.DEBUGGING,
                    ModelCapability.WEB_SEARCH
                ],
                "thinking_modes": ["minimal", "low", "medium", "high", "max"],
                "cost": "low",
                "speed": "fast",
                "context_window": 1000000,
                "description": "Best price-performance, adaptive thinking"
            },
            
            "gemini-2.5-pro": {
                "strengths": [
                    ModelCapability.DEEP_THINKING,
                    ModelCapability.ARCHITECTURE,
                    ModelCapability.SECURITY,
                    ModelCapability.CODE_REVIEW
                ],
                "thinking_modes": ["medium", "high", "max"],
                "cost": "high", 
                "speed": "slower",
                "context_window": 2000000,
                "description": "Most powerful thinking model, maximum accuracy"
            },
            
            "gpt-4o": {
                "strengths": [
                    ModelCapability.DEBUGGING,
                    ModelCapability.CODE_REVIEW,
                    ModelCapability.FAST_ANALYSIS
                ],
                "thinking_modes": None,  # No thinking modes for GPT
                "cost": "medium",
                "speed": "medium", 
                "context_window": 128000,
                "description": "Strong reasoning and general intelligence"
            }
        }
        
        # Tool to capability mapping
        self.tool_capabilities = {
            "analyze": ModelCapability.FAST_ANALYSIS,
            "codereview": ModelCapability.CODE_REVIEW,
            "debug": ModelCapability.DEBUGGING,
            "thinkdeep": ModelCapability.DEEP_THINKING,
            "websearch": ModelCapability.WEB_SEARCH
        }
        
        # Complexity-based model preferences (updated for 2.5 models)
        self.complexity_preferences = {
            "simple": ["gemini-2.5-flash", "gpt-4o"],
            "moderate": ["gemini-2.5-flash", "gemini-2.5-pro", "gpt-4o"],
            "complex": ["gemini-2.5-pro", "gemini-2.5-flash"],
            "specialized": ["gemini-2.5-pro"]
        }

    def select_model_for_consultation(
        self, 
        tool: str, 
        complexity: str = "moderate",
        context_keywords: List[str] = None,
        available_models: List[str] = None,
        prefer_speed: bool = False
    ) -> str:
        """
        Select the best model for a consultation based on multiple factors
        
        Args:
            tool: The consultation tool being used
            complexity: Task complexity (simple, moderate, complex, specialized)
            context_keywords: Keywords from the task context
            available_models: List of available models to choose from
            prefer_speed: Whether to prioritize speed over accuracy
            
        Returns:
            str: The recommended model name
        """
        context_keywords = context_keywords or []
        available_models = available_models or list(self.model_capabilities.keys())
        
        # Get required capability for this tool
        required_capability = self.tool_capabilities.get(tool, ModelCapability.FAST_ANALYSIS)
        
        # Start with complexity-based preferences
        complexity_models = self.complexity_preferences.get(complexity, ["gemini-2.5-flash"])
        
        # Filter by available models
        candidate_models = [m for m in complexity_models if m in available_models]
        
        if not candidate_models:
            candidate_models = available_models
        
        # Score models based on multiple factors
        model_scores = {}
        
        for model in candidate_models:
            score = 0
            capabilities = self.model_capabilities[model]
            
            # Base capability match
            if required_capability in capabilities["strengths"]:
                score += 3
                
            # Context keyword bonuses
            score += self._calculate_keyword_bonus(model, context_keywords)
            
            # Speed preference
            if prefer_speed:
                if capabilities["speed"] == "fast":
                    score += 2
                elif capabilities["speed"] == "medium":
                    score += 1
            else:
                # Prefer accuracy for non-speed-critical tasks
                if capabilities["cost"] == "high":  # Usually means more capable
                    score += 1
            
            # Tool-specific preferences
            score += self._get_tool_specific_bonus(tool, model)
            
            model_scores[model] = score
        
        # Return highest scoring model
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        return best_model

    def select_thinking_mode(
        self, 
        model: str, 
        tool: str, 
        complexity: str = "moderate",
        prefer_speed: bool = False
    ) -> Optional[str]:
        """
        Select appropriate thinking mode for the model and task
        
        Args:
            model: The model being used
            tool: The consultation tool
            complexity: Task complexity
            prefer_speed: Whether to prioritize speed
            
        Returns:
            Optional[str]: Recommended thinking mode or None
        """
        capabilities = self.model_capabilities.get(model, {})
        thinking_modes = capabilities.get("thinking_modes")
        
        if not thinking_modes:
            return None
        
        # Tool-specific thinking mode preferences
        tool_thinking_preferences = {
            "analyze": "medium",
            "codereview": "medium", 
            "debug": "high",
            "thinkdeep": "max",
            "websearch": "low"
        }
        
        # Complexity adjustments
        complexity_adjustments = {
            "simple": -1,
            "moderate": 0,
            "complex": +1,
            "specialized": +2
        }
        
        # Start with tool preference
        base_mode = tool_thinking_preferences.get(tool, "medium")
        
        # Get numeric level
        mode_levels = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "max": 4}
        level_modes = {v: k for k, v in mode_levels.items()}
        
        current_level = mode_levels.get(base_mode, 2)
        
        # Apply complexity adjustment
        adjustment = complexity_adjustments.get(complexity, 0)
        new_level = max(0, min(4, current_level + adjustment))
        
        # Apply speed preference
        if prefer_speed:
            new_level = max(0, new_level - 1)
        
        # Ensure the mode is available for this model
        target_mode = level_modes[new_level]
        if target_mode in thinking_modes:
            return target_mode
        
        # Fallback to closest available mode
        available_levels = [mode_levels[mode] for mode in thinking_modes if mode in mode_levels]
        if available_levels:
            closest_level = min(available_levels, key=lambda x: abs(x - new_level))
            return level_modes[closest_level]
        
        return None

    def _calculate_keyword_bonus(self, model: str, keywords: List[str]) -> int:
        """Calculate bonus score based on context keywords"""
        keyword_model_bonuses = {
            "security": {"gemini-2.5-pro": 2, "gemini-2.5-flash": 1},
            "performance": {"gpt-4o": 2, "gemini-2.5-flash": 1},
            "architecture": {"gemini-2.5-pro": 2},
            "debugging": {"gpt-4o": 2, "gemini-2.5-flash": 1},
            "analysis": {"gemini-2.5-pro": 1, "gemini-2.5-flash": 1},
            "thinking": {"gemini-2.5-pro": 3, "gemini-2.5-flash": 2},  # Gemini 2.5 has adaptive thinking
            "complex": {"gemini-2.5-pro": 2},
            "multimodal": {"gemini-2.5-pro": 2, "gemini-2.5-flash": 2}  # Gemini 2.5 supports audio/video
        }
        
        total_bonus = 0
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for key_pattern, bonuses in keyword_model_bonuses.items():
                if key_pattern in keyword_lower:
                    total_bonus += bonuses.get(model, 0)
        
        return total_bonus

    def _get_tool_specific_bonus(self, tool: str, model: str) -> int:
        """Get tool-specific model bonuses"""
        tool_bonuses = {
            "thinkdeep": {"gemini-2.5-pro": 3},  # 2.5 Pro excels at thinking
            "debug": {"gpt-4o": 2, "gemini-2.5-flash": 1},
            "codereview": {"gemini-2.5-pro": 2, "gemini-2.5-flash": 1},
            "analyze": {"gemini-2.5-flash": 2, "gemini-2.5-pro": 1},  # Flash is optimized for this
            "websearch": {"gemini-2.5-flash": 3}  # Fast and efficient
        }
        
        return tool_bonuses.get(tool, {}).get(model, 0)

    def get_model_recommendations(self, task_description: str) -> Dict[str, str]:
        """
        Get model recommendations for different aspects of a task
        
        Args:
            task_description: Description of the task
            
        Returns:
            Dict mapping aspects to recommended models
        """
        keywords = task_description.lower().split()
        
        recommendations = {}
        
        # Analyze task characteristics
        if any(word in task_description.lower() for word in ['security', 'vulnerability', 'attack']):
            recommendations['security'] = self.select_model_for_consultation("codereview", "complex", keywords)
        
        if any(word in task_description.lower() for word in ['debug', 'error', 'bug', 'issue']):
            recommendations['debugging'] = self.select_model_for_consultation("debug", "moderate", keywords)
        
        if any(word in task_description.lower() for word in ['architecture', 'design', 'system']):
            recommendations['architecture'] = self.select_model_for_consultation("thinkdeep", "complex", keywords)
        
        if any(word in task_description.lower() for word in ['performance', 'optimize', 'speed']):
            recommendations['performance'] = self.select_model_for_consultation("analyze", "moderate", keywords)
        
        if any(word in task_description.lower() for word in ['think', 'reason', 'complex', 'analysis']):
            recommendations['thinking'] = self.select_model_for_consultation("thinkdeep", "specialized", keywords)
        
        # Default analysis recommendation
        if not recommendations:
            recommendations['general'] = self.select_model_for_consultation("analyze", "moderate", keywords)
        
        return recommendations

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        return self.model_capabilities.get(model, {
            "description": "Unknown model",
            "strengths": [],
            "thinking_modes": None,
            "cost": "unknown",
            "speed": "unknown",
            "context_window": 0
        })
