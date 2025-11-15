from typing import Dict, Optional


class ObjectiveFocusManager:
    """
    Maintains a concise, high-signal 'current focus' string to keep actions aligned
    with story progression and recently learned hints (e.g., from NPC dialogue).
    """
    def __init__(self):
        self._current_focus: str = ""

    def update_from_extraction(self, extraction: Dict[str, str]) -> None:
        """
        Update current focus using structured extraction from dialogue.
        Expected keys: 'direction', 'location'
        """
        direction = (extraction.get("direction") or "").strip()
        location = (extraction.get("location") or "").strip()
        if location and direction:
            self._current_focus = f"Head {direction} towards {location}"
        elif location:
            self._current_focus = f"Go to {location}"
        elif direction:
            self._current_focus = f"Move {direction}"

    def update_from_text(self, text: str) -> None:
        """
        Opportunistic update from unstructured text (e.g., LLM output field).
        Keep very conservative to avoid noise.
        """
        if not text:
            return
        lower = text.lower()
        # Minimal heuristic: look for 'go to X' or 'head to X'
        triggers = ["go to ", "head to ", "towards "]
        if any(t in lower for t in triggers):
            # Keep the text short as-is to avoid mis-parsing
            self._current_focus = text[:120]

    def get_current_focus(self, fallback_plan: Optional[str] = "") -> str:
        """
        Return the best current focus string for downstream prompts.
        Preference order: dialogue-derived focus > previous plan > generic default.
        """
        if self._current_focus:
            return self._current_focus
        if fallback_plan:
            return fallback_plan
        return "Progress towards the first gym (Petalburg -> Rustboro)"  # generic guiding default


