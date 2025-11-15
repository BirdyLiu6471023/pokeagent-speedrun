import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


MEMORY_CACHE_DIR = ".pokeagent_cache"
MEMORY_FILE = "agent_memory.json"


@dataclass
class LTObjective:
	id: str
	description: str
	objective_type: str
	target_value: Optional[Any] = None
	completed: bool = False
	created_at: str = field(default_factory=lambda: datetime.now().isoformat())
	completed_at: Optional[str] = None
	notes: str = ""


@dataclass
class LongTermMemory:
	active_objectives: List[LTObjective] = field(default_factory=list)
	completed_objectives: List[LTObjective] = field(default_factory=list)
	visited_locations: Dict[str, int] = field(default_factory=dict)  # map_name -> visits
	visited_coords: Dict[str, int] = field(default_factory=dict)     # "map:x,y" -> visits
	last_focus: str = ""
	dialogue_hints: List[str] = field(default_factory=list)
	npc_tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # key -> {description, source, created_at, completed, completed_at, notes}

	_loaded_once: bool = field(default=False, init=False, repr=False)

	def _path(self) -> str:
		os.makedirs(MEMORY_CACHE_DIR, exist_ok=True)
		return os.path.join(MEMORY_CACHE_DIR, MEMORY_FILE)

	def load(self) -> None:
		if self._loaded_once:
			return
		self._loaded_once = True
		try:
			path = self._path()
			if not os.path.exists(path):
				return
			with open(path, "r", encoding="utf-8") as f:
				raw = json.load(f)
			self.active_objectives = [LTObjective(**o) for o in raw.get("active_objectives", [])]
			self.completed_objectives = [LTObjective(**o) for o in raw.get("completed_objectives", [])]
			self.visited_locations = dict(raw.get("visited_locations", {}))
			self.visited_coords = dict(raw.get("visited_coords", {}))
			self.last_focus = raw.get("last_focus", "")
			self.dialogue_hints = list(raw.get("dialogue_hints", []))[-10:]
			self.npc_tasks = dict(raw.get("npc_tasks", {}))
		except Exception:
			# Non-fatal
			pass

	def save(self) -> None:
		try:
			payload = {
				"active_objectives": [asdict(o) for o in self.active_objectives][-20:],
				"completed_objectives": [asdict(o) for o in self.completed_objectives][-50:],
				"visited_locations": self.visited_locations,
				"visited_coords": self.visited_coords,
				"last_focus": self.last_focus[:200],
				"dialogue_hints": self.dialogue_hints[-10:],
				"npc_tasks": self.npc_tasks,
			}
			with open(self._path(), "w", encoding="utf-8") as f:
				json.dump(payload, f, ensure_ascii=False, indent=2)
		except Exception:
			pass

	def mark_visit(self, map_name: Optional[str], coords: Optional[Tuple[int, int]]) -> None:
		if map_name:
			self.visited_locations[map_name] = self.visited_locations.get(map_name, 0) + 1
		if map_name and coords:
			key = f"{map_name}:{coords[0]},{coords[1]}"
			self.visited_coords[key] = self.visited_coords.get(key, 0) + 1

	def set_last_focus(self, focus: str) -> None:
		if focus and focus.strip():
			self.last_focus = focus.strip()[:200]

	def add_dialogue_hint(self, text: str) -> None:
		text = (text or "").strip()
		if not text:
			return
		self.dialogue_hints.append(text[:200])
		self.dialogue_hints = self.dialogue_hints[-10:]

	def add_objective(self, description: str, objective_type: str, target_value: Any = None, obj_id: Optional[str] = None) -> str:
		oid = obj_id or f"obj_{int(datetime.now().timestamp())}"
		self.active_objectives.append(LTObjective(id=oid, description=description, objective_type=objective_type, target_value=target_value))
		return oid

	def complete_objective(self, obj_id: str, notes: str = "") -> bool:
		for i, o in enumerate(self.active_objectives):
			if o.id == obj_id and not o.completed:
				o.completed = True
				o.completed_at = datetime.now().isoformat()
				o.notes = notes
				self.completed_objectives.append(o)
				del self.active_objectives[i]
				return True
		return False
	
	def npc_upsert_task(self, key: str, description: str, source_npc: str = "", notes: str = "") -> None:
		key = (key or "").strip().lower()
		if not key:
			return
		now = datetime.now().isoformat()
		if key not in self.npc_tasks:
			self.npc_tasks[key] = {
				"description": description,
				"source_npc": source_npc,
				"created_at": now,
				"completed": False,
				"completed_at": None,
				"notes": notes,
			}
		else:
			# Update description/source if provided
			if description:
				self.npc_tasks[key]["description"] = description
			if source_npc:
				self.npc_tasks[key]["source_npc"] = source_npc
			if notes:
				self.npc_tasks[key]["notes"] = (self.npc_tasks[key].get("notes") or "") + f" | {notes}"
	
	def npc_mark_completed(self, key: str, notes: str = "") -> bool:
		key = (key or "").strip().lower()
		if not key or key not in self.npc_tasks:
			return False
		if not self.npc_tasks[key].get("completed"):
			self.npc_tasks[key]["completed"] = True
			self.npc_tasks[key]["completed_at"] = datetime.now().isoformat()
			if notes:
				self.npc_tasks[key]["notes"] = (self.npc_tasks[key].get("notes") or "") + f" | {notes}"
		return True

	def build_summary(self) -> str:
		parts: List[str] = []
		if self.last_focus:
			parts.append(f"FOCUS: {self.last_focus}")
		if self.active_objectives:
			objs = [f"[{o.objective_type}] {o.description}" for o in self.active_objectives[:5]]
			parts.append("ACTIVE: " + " | ".join(objs))
		if self.completed_objectives:
			done = [o.description for o in self.completed_objectives[-3:]]
			parts.append("DONE: " + " | ".join(done))
		if self.dialogue_hints:
			parts.append("HINTS: " + " | ".join(self.dialogue_hints[-3:]))
		# NPC tasks summary
		if self.npc_tasks:
			pending = [k for k, v in self.npc_tasks.items() if not v.get("completed")]
			done = [k for k, v in self.npc_tasks.items() if v.get("completed")]
			if pending:
				parts.append("NPC_TASKS_PENDING: " + ", ".join(pending[:3]))
			if done:
				parts.append("NPC_TASKS_DONE: " + ", ".join(done[-2:]))
		# visited summary
		if self.visited_locations:
			top = sorted(self.visited_locations.items(), key=lambda kv: kv[1], reverse=True)[:2]
			vis = ", ".join([f"{k}({v})" for k, v in top])
			parts.append(f"VISITED: {vis}")
		return " || ".join(parts) if parts else "No prior memory."


_GLOBAL_MEMORY: Optional[LongTermMemory] = None


def get_global_memory() -> LongTermMemory:
	global _GLOBAL_MEMORY
	if _GLOBAL_MEMORY is None:
		_GLOBAL_MEMORY = LongTermMemory()
		_GLOBAL_MEMORY.load()
	return _GLOBAL_MEMORY


