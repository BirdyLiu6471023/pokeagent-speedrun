from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
logger = logging.getLogger(__name__)

def get_game_context(game_state: Dict[str, Any]) -> str:
    """Determine current game context (overworld, battle, menu, dialogue)"""
    try:
        # Check if in battle
        is_in_battle = game_state.get("game", {}).get("is_in_battle", False)
        if is_in_battle:
            logger.debug(f"Detected battle context")
            return "battle"
        
        # Check if dialogue is active
        dialogue_state = game_state.get("game", {}).get("dialogue", {})
        if dialogue_state.get("active", False) or dialogue_state.get("text", "").strip():
            return "dialogue"
        
        # Check if in menu (simplified detection)
        # Could be enhanced with more sophisticated menu detection
        player_state = game_state.get("player", {})
        if player_state.get("in_menu", False):
            return "menu"
        
        # Default to overworld
        return "overworld"
        
    except Exception as e:
        logger.warning(f"Error determining game context: {e}")
        return "unknown"


def get_map_id(self, game_state: Dict[str, Any]) -> Optional[int]:
    """Extract map ID from game state"""
    try:
        return game_state.get("map", {}).get("id")
    except Exception as e:
        logger.warning(f"Error getting map ID: {e}")
    return None