from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from collections import Counter
from io import BytesIO
from PIL import Image
import base64
import numpy as np
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


def get_map_id(game_state: Dict[str, Any]) -> Optional[int]:
    """Extract map ID from game state"""
    try:
        # Preferred: state["map"]["stitched_map_info"]["current_area"]["id"] (hex str)
        map_section = game_state.get("map", {})
        stitched = map_section.get("stitched_map_info") or {}
        current_area = stitched.get("current_area") or {}
        area_id = current_area.get("id")
        if area_id is not None:
            try:
                if isinstance(area_id, str):
                    return int(area_id, 16)
                if isinstance(area_id, int):
                    return area_id
            except ValueError:
                pass

        # Next: state["map"]["id"] as int
        map_section = game_state.get("map", {})
        map_id = map_section.get("id")
        if isinstance(map_id, int):
            return map_id

        # Also support state["map"]["current_area"]["id"] (if present)
        current_area = map_section.get("current_area") or {}
        area_id = current_area.get("id")
        if area_id is not None:
            try:
                # Accept hex string or int
                if isinstance(area_id, str):
                    return int(area_id, 16)
                if isinstance(area_id, int):
                    return area_id
            except ValueError:
                pass

        # Backward-compat: some responses used 'mapp'
        legacy_map = game_state.get("mapp", {})
        legacy_id = legacy_map.get("id")
        if isinstance(legacy_id, int):
            return legacy_id

        # If map bank/number are present anywhere, combine them
        bank = map_section.get("bank") or game_state.get("map_bank")
        number = map_section.get("number") or game_state.get("map_number")
        if isinstance(bank, int) and isinstance(number, int):
            return (bank << 8) | number
    except Exception as e:
        logger.warning(f"Error getting map ID: {e}")
        return None

def get_player_coords(game_state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Extract player coordinates from game state"""
        try:
            player = game_state.get("player", {})
            # Try position.x/y first (standard format)
            position = player.get("position", {})
            if position:
                x = position.get("x")
                y = position.get("y")
                if x is not None and y is not None:
                    return (x, y)
            
            # Fallback: try direct x/y on player
            x = player.get("x")
            y = player.get("y")
            if x is not None and y is not None:
                return (x, y)
        except Exception as e:
            logger.warning(f"Error getting player coords: {e}")
        return None

def get_player_facing(game_state: Dict[str, Any]) -> Optional[str]:
    """
    Best-effort extraction of the player's facing direction from the provided game_state.
    
    Returns one of: "North", "South", "East", "West" when available, otherwise None.
    
    Supported sources (in order of precedence):
    - game_state["player"]["facing"]
    - game_state["map"]["player_facing"]
    - game_state["map"]["current_area"]["player_facing"]
    - game_state["map"]["stitched_map_info"]["current_area"]["player_facing"]
    
    Accepted input formats:
    - String directions: "North"/"South"/"East"/"West" (case-insensitive)
    - Abbreviations: "N","S","E","W"
    - UI terms: "up","down","left","right"
    - Integers using emulator mapping: 0=South, 1=North, 2=West, 3=East
    """
    def _normalize_facing(raw_value: Any) -> Optional[str]:
        if raw_value is None:
            return None
        # Integer mapping from emulator memory (see memory_reader.SAVESTATE_PLAYER_FACING_OFFSET)
        if isinstance(raw_value, int):
            mapping = {0: "South", 1: "North", 2: "West", 3: "East"}
            return mapping.get(raw_value)
        # Try to parse string-like inputs
        try:
            text = str(raw_value).strip().lower()
        except Exception:
            return None
        if not text:
            return None
        direct_map = {
            "north": "North",
            "south": "South",
            "east": "East",
            "west": "West",
            "n": "North",
            "s": "South",
            "e": "East",
            "w": "West",
            "up": "North",
            "down": "South",
            "left": "West",
            "right": "East",
        }
        return direct_map.get(text)
    
    # Candidate sources in priority order
    candidates = []
    try:
        candidates.append(game_state.get("player", {}).get("facing"))
    except Exception:
        pass
    try:
        map_section = game_state.get("map", {}) or {}
        candidates.append(map_section.get("player_facing"))
        candidates.append((map_section.get("current_area") or {}).get("player_facing"))
        stitched = map_section.get("stitched_map_info") or {}
        candidates.append((stitched.get("current_area") or {}).get("player_facing"))
    except Exception:
        pass
    
    for candidate in candidates:
        normalized = _normalize_facing(candidate)
        if normalized:
            return normalized
    
    # Unknown if not present in state
    return None

def get_map_name(game_state: Dict[str, Any]) -> Optional[str]:
    """Extract a human-readable location name from the game state.

    Prefers map.stitched_map_info.current_area.name, then player.location, then map.location.
    """
    try:
        map_section = game_state.get("map", {})
        stitched = map_section.get("stitched_map_info") or {}
        current_area = stitched.get("current_area") or {}
        name = current_area.get("name")
        if isinstance(name, str) and name.strip():
            return name

        player_loc = game_state.get("player", {}).get("location")
        if isinstance(player_loc, str) and player_loc.strip():
            return player_loc

        map_loc = map_section.get("location")
        if isinstance(map_loc, str) and map_loc.strip():
            return map_loc
    except Exception as e:
        logger.warning(f"Error getting map name: {e}")
    return None

def get_current_map_grid(game_state: Dict[str, Any]) -> Dict[Tuple[int, int], str]:
    """Return a simplified grid for the current location using MapStitcher.

    Keys are (x, y) tuples in a compacted, relative coordinate space of the explored area.
    Values are simplified symbols ('.', '#', '~', 'W', 'D', 'S', arrows, etc.).
    """
    try:
        from utils.map_stitcher import MapStitcher  # Local import to avoid heavy dependency at module import
    except Exception as e:
        logger.warning(f"MapStitcher import failed: {e}")
        return {}

    location_name = get_map_name(game_state)
    if not location_name:
        return {}

    try:
        map_stitcher = MapStitcher()
        grid = map_stitcher.get_location_grid(location_name, simplified=True)
        return grid or {}
    except Exception as e:
        logger.warning(f"Error generating map grid for '{location_name}': {e}")
        return {}

def get_current_map_display(game_state: Dict[str, Any]) -> Optional[str]:
    """Return an ASCII map display of the current location in the same format
    produced by MapStitcher.generate_location_map_display (e.g., lines like the ROUTE 102 map).

    If MapStitcher has no accumulated data yet, falls back to formatting the 15x15
    local memory tiles when available.
    """
    try:
        from utils.map_stitcher import MapStitcher  # Deferred import
    except Exception as e:
        logger.warning(f"MapStitcher import failed: {e}")
        return None

    map_section = game_state.get("map", {}) or {}
    location_name = get_map_name(game_state) or map_section.get("location") or "Unknown"

    # Player position (world coords)
    player_pos = get_player_coords(game_state)

    # Optional NPCs and stitched connections
    npcs = map_section.get("object_events") or []
    stitched = map_section.get("stitched_map_info") or {}
    current_area = stitched.get("current_area") or {}
    connections = current_area.get("connections") or []

    try:
        map_stitcher = MapStitcher()
        lines = map_stitcher.generate_location_map_display(
            location_name=location_name,
            player_pos=player_pos,
            npcs=npcs,
            connections=connections,
        )
        if lines:
            return "\n".join(lines)
    except Exception as e:
        logger.warning(f"MapStitcher display generation failed: {e}")

    # Fallback: format from local 15x15 memory tiles if available
    try:
        tiles = map_section.get("tiles")
        if tiles:
            from utils.map_formatter import (
                format_map_for_llm,
                format_map_grid,
                generate_dynamic_legend,
            )
            facing = "South"
            player_coords_dict = None
            if player_pos is not None:
                player_coords_dict = {"x": player_pos[0], "y": player_pos[1]}
            map_display = format_map_for_llm(tiles, facing, npcs, player_coords_dict)
            grid = format_map_grid(tiles, facing, npcs, player_coords_dict)
            legend = generate_dynamic_legend(grid)
            header = f"\n--- MAP: {str(location_name).upper()} (from memory) ---"
            return f"{header}\n{map_display}\n\n{legend}"
    except Exception as e:
        logger.warning(f"Local tiles fallback failed: {e}")

    return None

def detect_stuck_pattern(coords: Optional[Tuple[int, int]], context: str, game_state: Dict[str, Any] = None) -> bool:
    """Detect if the agent appears to be stuck in a location/context"""
    if not coords:
        return False
    
    # Don't trigger stuck detection during contexts where staying in place is expected
    if context in ["battle", "dialogue", "menu"]:
        logger.debug(f"Skipping stuck detection - context: {context}")
        return False
    
    # Check for title sequence if game state is available
    if game_state:
        # Check if in title sequence (no player name or invalid coordinates)
        player_name = game_state.get("player", {}).get("name", "").strip()
        if not player_name or player_name == "????????":
            return False
            
        # Check if game state indicates title/intro
        game_state_value = game_state.get("game", {}).get("game_state", "").lower()
        if "title" in game_state_value or "intro" in game_state_value:
            return False
        
    ### RETURN????? 

def get_not_suggested_directions(town_name, pos, map_memory):
    '''
    pos is a tuple (x, y) which is th
    this is a function to provide suggestion for movement. 
    If current objective is not exploring the map (like interacting with a NPC),
    then this function would not provide any suggestions.
    '''
    x,y = pos 
    suggest_not_directions = []

    if (x-1, y) in map_memory[town_name]:
        suggest_not_directions.append("LEFT")
    if (x+1, y) in map_memory[town_name]:
        suggest_not_directions.append("RIGHT")
    if (x, y+1) in map_memory[town_name]:
        suggest_not_directions.append("DOWN")
    if (x, y-1) in map_memory[town_name]:
        suggest_not_directions.append("UP")
    
    return suggest_not_directions


def _to_data_url(image_like):
    """Convert PIL image or numpy array to data URL for OpenAI vision messages."""
    if image_like is None:
        return None
    if isinstance(image_like, np.ndarray):
        pil = Image.fromarray(image_like)
    elif isinstance(image_like, Image.Image):
        pil = image_like
    else:
        return None
    buff = BytesIO()
    pil.save(buff, format="PNG")
    b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _decode_base64_image(b64_str: str) -> Optional[Image.Image]:
    try:
        if not b64_str:
            return None
        data = base64.b64decode(b64_str)
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None


def extract_frame_from_game_state(game_state: dict) -> Optional[Union[Image.Image, np.ndarray]]:
    """Extract a frame image (PIL or numpy array) from a heterogeneous game_state structure.

    Supports keys:
    - 'frame' (PIL/numpy or base64 str)
    - 'screenshot' (PIL/numpy)
    - 'visual': {'screenshot': PIL/numpy, 'screenshot_base64': str}
    """
    if not game_state:
        return None

    # Direct frame
    frame = game_state.get('frame')
    if isinstance(frame, (Image.Image, np.ndarray)):
        return frame
    if isinstance(frame, str):
        decoded = _decode_base64_image(frame)
        if decoded is not None:
            return decoded

    # Top-level screenshot
    screenshot = game_state.get('screenshot')
    if isinstance(screenshot, (Image.Image, np.ndarray)):
        return screenshot

    # Visual section
    visual = game_state.get('visual') or {}
    vis_img = visual.get('screenshot')
    if isinstance(vis_img, (Image.Image, np.ndarray)):
        return vis_img
    vis_b64 = visual.get('screenshot_base64')
    if isinstance(vis_b64, str):
        decoded = _decode_base64_image(vis_b64)
        if decoded is not None:
            return decoded

    return None

def _initialize_storyline_objectives(self):
    """Initialize the main storyline objectives for Pokémon Emerald progression"""
    storyline_objectives = [
        {
            "id": "story_game_start",
            "description": "Complete title sequence and begin the game",
            "objective_type": "system",
            "target_value": "Game Running",
            "milestone_id": "GAME_RUNNING"
        },
        {
            "id": "story_littleroot_town",
            "description": "Arrive in Littleroot Town and explore the area",
            "objective_type": "location", 
            "target_value": "Littleroot Town",
            "milestone_id": "LITTLEROOT_TOWN"
        },
        {
            "id": "story_route_101",
            "description": "Travel north to Route 101 and encounter Prof. Birch",
            "objective_type": "location",
            "target_value": "Route 101", 
            "milestone_id": "ROUTE_101"
        },
        {
            "id": "story_starter_chosen",
            "description": "Choose starter Pokémon and receive first party member",
            "objective_type": "pokemon",
            "target_value": "Starter Pokémon",
            "milestone_id": "STARTER_CHOSEN"
        },
        {
            "id": "story_oldale_town",
            "description": "Continue journey to Oldale Town",
            "objective_type": "location",
            "target_value": "Oldale Town",
            "milestone_id": "OLDALE_TOWN"
        },
        {
            "id": "story_route_103",
            "description": "Travel to Route 103 to meet rival",
            "objective_type": "location",
            "target_value": "Route 103",
            "milestone_id": "ROUTE_103"
        },
        {
            "id": "story_route_102",
            "description": "Return through Route 102 toward Petalburg City",
            "objective_type": "location",
            "target_value": "Route 102", 
            "milestone_id": "ROUTE_102"
        },
        {
            "id": "story_petalburg_city",
            "description": "Navigate to Petalburg City and visit Dad's gym",
            "objective_type": "location",
            "target_value": "Petalburg City",
            "milestone_id": "PETALBURG_CITY"
        },
        {
            "id": "story_route_104",
            "description": "Travel north through Route 104 toward Petalburg Woods",
            "objective_type": "location",
            "target_value": "Route 104",
            "milestone_id": "ROUTE_104"
        },
        {
            "id": "story_petalburg_woods",
            "description": "Navigate through Petalburg Woods to help Devon researcher",
            "objective_type": "location",
            "target_value": "Petalburg Woods",
            "milestone_id": "PETALBURG_WOODS"
        },
        {
            "id": "story_rustboro_city",
            "description": "Arrive in Rustboro City and deliver Devon Goods",
            "objective_type": "location",
            "target_value": "Rustboro City",
            "milestone_id": "RUSTBORO_CITY"
        },
        {
            "id": "story_rustboro_gym",
            "description": "Enter the Rustboro Gym and prepare for Roxanne battle",
            "objective_type": "location",
            "target_value": "Rustboro Gym",
            "milestone_id": None  # Gym entry doesn't have separate milestone
        },
        {
            "id": "story_stone_badge",
            "description": "Defeat Roxanne and earn the Stone Badge",
            "objective_type": "battle",
            "target_value": "Stone Badge",
            "milestone_id": "STONE_BADGE"
        }
    ]
    
    # Add storyline objectives to the state
    for obj_data in storyline_objectives:
        objective = Objective(
            id=obj_data["id"],
            description=obj_data["description"],
            objective_type=obj_data["objective_type"],
            target_value=obj_data["target_value"],
            completed=False,
            progress_notes="Storyline objective - verified by emulator milestones",
            storyline=True,
            milestone_id=obj_data["milestone_id"]
        )
        self.state.objectives.append(objective)
        
    logger.info(f"Initialized {len(storyline_objectives)} storyline objectives for Emerald progression")
     