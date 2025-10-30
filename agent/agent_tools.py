from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, AIMessage 
import random
import logging
from utils.vlm import VLM
from utils.state_formatter import format_state_for_llm, format_state_summary, get_movement_options, get_party_health_summary
from agent.system_prompt import system_prompt
# Set up module logging
logger = logging.getLogger(__name__)

print("install successfully") 

def action_agent_node(agent_state, state_data, recent_actions, vlm, system_message, name):
    # Get formatted state context and useful summaries
    state_context = format_state_for_llm(state_data)
    state_summary = format_state_summary(state_data)
    movement_options = get_movement_options(state_data)
    party_health = get_party_health_summary(state_data)
    
    logger.info("[ACTION] Starting action decision")
    logger.info(f"[ACTION] State: {state_summary}")
    logger.info(f"[ACTION] Party health: {party_health['healthy_count']}/{party_health['total_count']} healthy")
    if movement_options:
        logger.info(f"[ACTION] Movement options: {movement_options}")
    
    # Build enhanced action context
    action_context = []
    
    # Extract key info for context
    game_data = state_data.get('game', {})
    
    # Battle vs Overworld context
    if game_data.get('in_battle', False):
        action_context.append("=== BATTLE MODE ===")
        battle_info = game_data.get('battle_info', {})
        if battle_info:
            if 'player_pokemon' in battle_info:
                player_pkmn = battle_info['player_pokemon']
                action_context.append(f"Your Pokemon: {player_pkmn.get('species_name', player_pkmn.get('species', 'Unknown'))} (Lv.{player_pkmn.get('level', '?')}) HP: {player_pkmn.get('current_hp', '?')}/{player_pkmn.get('max_hp', '?')}")
            if 'opponent_pokemon' in battle_info:
                opp_pkmn = battle_info['opponent_pokemon']
                action_context.append(f"Opponent: {opp_pkmn.get('species_name', opp_pkmn.get('species', 'Unknown'))} (Lv.{opp_pkmn.get('level', '?')}) HP: {opp_pkmn.get('current_hp', '?')}/{opp_pkmn.get('max_hp', '?')}")
    else:
        action_context.append("=== OVERWORLD MODE ===")
        
        # Movement options from utility
        if movement_options:
            action_context.append("Movement Options:")
            for direction, description in movement_options.items():
                action_context.append(f"  {direction}: {description}")
    
    # Party health summary
    if party_health['total_count'] > 0:
        action_context.append("=== PARTY STATUS ===")
        action_context.append(f"Healthy Pokemon: {party_health['healthy_count']}/{party_health['total_count']}")
        if party_health['critical_pokemon']:
            action_context.append("Critical Pokemon:")
            for critical in party_health['critical_pokemon']:
                action_context.append(f"  {critical}")
    
    # Recent actions context
    if recent_actions:
        action_context.append(f"Recent Actions: {', '.join(list(recent_actions)[-5:])}")
    
    context_str = "\n".join(action_context)

    action_prompt = f"""
    ★★★ COMPREHENSIVE GAME STATE DATA ★★★
    
    {state_context}
    
    ★★★ ENHANCED ACTION CONTEXT ★★★
    
    {context_str} 
    """ 

    prompt = ChatPromptTemplate.from_message(
        [
        ('system',
        "You are the agent playing Pokemon Emerald with a speedrunning mindset. Make quick, efficient decisions."
        "You need to make decision on which tool to use to get the best action based on current state and current target"
        "Current State Information: {action_prompt}"
        
        )

        ]

    )