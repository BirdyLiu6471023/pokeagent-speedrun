import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Optional, Tuple


def get_dialogue_text_from_state(game_state: dict) -> str:
    """
    Best-effort extraction of current dialogue text from heterogeneous game_state.
    Supports:
    - game.dialogue.text
    - game.textbox.text (fallback)
    - visual.ocr_text (fallback if present)
    """
    if not isinstance(game_state, dict):
        return ""
    try:
        game = game_state.get("game", {}) or {}
        dlg = game.get("dialogue", {}) or {}
        text = dlg.get("text") or ""
        if isinstance(text, str) and text.strip():
            return text.strip()
        # Fallbacks
        textbox = game.get("textbox", {}) or {}
        t2 = textbox.get("text") or ""
        if isinstance(t2, str) and t2.strip():
            return t2.strip()
        visual = game_state.get("visual", {}) or {}
        ocr = visual.get("ocr_text") or ""
        if isinstance(ocr, str) and ocr.strip():
            return ocr.strip()
    except Exception:
        pass
    return ""


@dataclass
class DialogueEntry:
    text: str
    location: str
    npc_name: str
    timestamp: datetime


class DialogueMemory:
    """
    Lightweight rolling memory of recent dialogue lines to surface important info to the agent.
    """
    def __init__(self, max_entries: int = 20):
        self._entries: Deque[DialogueEntry] = deque(maxlen=max_entries)

    def add_entry(self, text: str, location: str = "", npc_name: str = "") -> None:
        text = (text or "").strip()
        if not text:
            return
        self._entries.append(DialogueEntry(text=text, location=location or "", npc_name=npc_name or "", timestamp=datetime.now()))

    def summarize_recent(self, limit: int = 3) -> str:
        if not self._entries:
            return ""
        recent = list(self._entries)[-limit:]
        lines = []
        for e in recent:
            loc = f" @ {e.location}" if e.location else ""
            who = f"{e.npc_name}: " if e.npc_name else ""
            lines.append(f"{who}{e.text}{loc}")
        return " | ".join(lines)


_DIR_WORDS = {
    "north": "north",
    "south": "south",
    "east": "east",
    "west": "west",
    "up": "north",
    "down": "south",
    "left": "west",
    "right": "east",
}

_LOCATION_PATTERNS = [
    r"(?:to|towards|head to|go to|visit|reach)\s+(?P<loc>([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s(?:Town|City))",
    r"(?:to|towards|head to|go to|reach)\s+(?P<route>Route\s\d+)",
]


def extract_key_info_from_dialogue(text: str) -> Dict[str, str]:
    """
    Heuristic extraction for objective hints from dialogue text.
    Returns a dict with optional keys: 'direction', 'location', 'route', 'npc'.
    """
    result: Dict[str, str] = {}
    if not text:
        return result
    lower = text.lower()
    # Direction
    for k, v in _DIR_WORDS.items():
        if re.search(rf"\b{k}\b", lower):
            result["direction"] = v
            break
    # Locations
    for pat in _LOCATION_PATTERNS:
        m = re.search(pat, text)
        if m:
            if "loc" in m.groupdict() and m.group("loc"):
                result["location"] = m.group("loc")
                break
            if "route" in m.groupdict() and m.group("route"):
                result["location"] = m.group("route")
                break
    # NPC name (simple heuristic: leading "NAME:" prefix)
    npc_match = re.match(r"^\s*([A-Z][A-Z]+)\s*:\s*", text)
    if npc_match:
        result["npc"] = npc_match.group(1).title()
    return result


def build_dialogue_insight(memory: DialogueMemory, extraction: Dict[str, str]) -> str:
    """
    Build a concise insight string combining latest extraction and a short memory summary.
    """
    parts = []
    if extraction.get("location") and extraction.get("direction"):
        parts.append(f"Hint: go {extraction['direction']} towards {extraction['location']}.")
    elif extraction.get("location"):
        parts.append(f"Hint: head to {extraction['location']}.")
    elif extraction.get("direction"):
        parts.append(f"Hint: move {extraction['direction']}.")
    recent = memory.summarize_recent(limit=2)
    if recent:
        parts.append(f"Recent NPC info: {recent}")
    return " ".join(parts).strip()


def extract_npc_task_events(text: str) -> list:
    """
    Heuristic extraction of NPC task assignments/completions from dialogue text.
    Returns a list of events:
      { "action": "add"|"complete", "key": "set_clock", "description": "Set the bedroom clock", "source": "Mom" }
    """
    events = []
    if not text:
        return events
    lower = text.lower()
    # Detect "set the clock" assignment by Mom or generic instruction
    if ("set" in lower and "clock" in lower and ("please" in lower or "your room" in lower or "bedroom" in lower)):
        events.append({
            "action": "add",
            "key": "set_clock",
            "description": "Set the bedroom clock",
            "source": "Mom"
        })
    # Detect completion acknowledgements
    if ("clock" in lower) and any(kw in lower for kw in ["set", "adjusted", "done", "looks good", "thank"]):
        events.append({
            "action": "complete",
            "key": "set_clock",
            "description": "Set the bedroom clock",
            "source": "Mom"
        })
    return events


