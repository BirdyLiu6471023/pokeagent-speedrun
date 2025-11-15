from langchain_core.tools import tool
import logging
from typing import List, Tuple, Optional, Dict

# Set up module logging
logger = logging.getLogger(__name__)

################################################################################
# Helper functions for map parsing and simple pathing (text-based ASCII maps)
################################################################################

def _parse_map_display(map_display: str) -> Tuple[List[List[str]], Optional[Tuple[int, int]]]:
	"""
	Parse an ASCII map display into a grid and find the player's position 'P'.
	Returns (grid, player_pos) where grid[y][x] is a single-character tile.
	Player position is (x, y) or None if not found.
	"""
	if not isinstance(map_display, str) or not map_display.strip():
		return [], None
	lines = [line.rstrip("\n") for line in map_display.splitlines() if line.strip() != ""]
	if not lines:
		return [], None
	grid: List[List[str]] = [list(row) for row in lines]
	player_pos: Optional[Tuple[int, int]] = None
	for y, row in enumerate(grid):
		for x, ch in enumerate(row):
			if ch == "P":
				player_pos = (x, y)
				break
		if player_pos:
			break
	return grid, player_pos


def _is_passable(ch: str) -> bool:
	"""
	Heuristic passability for ASCII map tiles.
	- Walls: '#'
	- Everything else considered passable for one-step suggestion, including '.', ' ', '?', 'D', 'S'
	"""
	if ch == "#":
		return False
	return True


def _neighbors(x: int, y: int, grid: List[List[str]]) -> List[Tuple[int, int, str]]:
	"""
	Return passable 4-neighbors and the direction name to step there from (x,y).
	"""
	h = len(grid)
	w = len(grid[0]) if h > 0 else 0
	out: List[Tuple[int, int, str]] = []
	candidates = [
		(0, -1, "UP"),
		(0, 1, "DOWN"),
		(-1, 0, "LEFT"),
		(1, 0, "RIGHT"),
	]
	for dx, dy, dname in candidates:
		nx, ny = x + dx, y + dy
		if 0 <= nx < w and 0 <= ny < h and _is_passable(grid[ny][nx]):
			out.append((nx, ny, dname))
	return out


def _bfs_first_step_towards_targets(grid: List[List[str]], start: Tuple[int, int], targets: List[str]) -> Optional[str]:
	"""
	BFS to nearest tile whose character is in 'targets'. Returns the FIRST step direction
	from start towards the found target, or None if not found. Only passable tiles are traversed.
	"""
	from collections import deque
	sx, sy = start
	h = len(grid)
	w = len(grid[0]) if h > 0 else 0
	visited = [[False] * w for _ in range(h)]
	queue = deque()
	# Keep track of the first step direction for each enqueued node
	for nx, ny, dname in _neighbors(sx, sy, grid):
		queue.append((nx, ny, dname))
		visited[ny][nx] = True
	# If start is already on a target, just return a safe no-op (None -> let LLM decide)
	if grid[sy][sx] in targets:
		return None
	while queue:
		x, y, first_step = queue.popleft()
		if grid[y][x] in targets:
			return first_step
		for nx, ny, _d in _neighbors(x, y, grid):
			if not visited[ny][nx]:
				visited[ny][nx] = True
				queue.append((nx, ny, first_step))
	return None


def _bfs_first_step_to_coordinates(grid: List[List[str]], start: Tuple[int, int], targets: set) -> Optional[str]:
	"""
	BFS to nearest coordinate in 'targets' (set of (x,y)). Returns the FIRST step direction.
	Traverses only passable tiles.
	"""
	from collections import deque
	sx, sy = start
	h = len(grid)
	w = len(grid[0]) if h > 0 else 0
	visited = [[False] * w for _ in range(h)]
	queue = deque()
	for nx, ny, dname in _neighbors(sx, sy, grid):
		queue.append((nx, ny, dname))
		visited[ny][nx] = True
	if (sx, sy) in targets:
		return None
	while queue:
		x, y, first_step = queue.popleft()
		if (x, y) in targets:
			return first_step
		for nx, ny, _d in _neighbors(x, y, grid):
			if not visited[ny][nx]:
				visited[ny][nx] = True
				queue.append((nx, ny, first_step))
	return None


def _collect_frontier_approach_tiles_near_stairs(grid: List[List[str]]) -> List[Tuple[int, int]]:
	"""
	Identify '#' tiles that are adjacent to 'S' (stairs). For each such '#',
	collect its passable neighbors as 'approach tiles' to explore that occluded area.
	We don't step into '#', but we guide the agent adjacent to it.
	"""
	h = len(grid)
	w = len(grid[0]) if h > 0 else 0
	approach: List[Tuple[int, int]] = []
	def inb(x: int, y: int) -> bool:
		return 0 <= x < w and 0 <= y < h
	for y in range(h):
		for x in range(w):
			if grid[y][x] != "#":
				continue
			# Check if this '#' touches any 'S'
			adj_s = False
			for dx, dy, _ in [(0, -1, "UP"), (0, 1, "DOWN"), (-1, 0, "LEFT"), (1, 0, "RIGHT")]:
				nx, ny = x + dx, y + dy
				if inb(nx, ny) and grid[ny][nx] == "S":
					adj_s = True
					break
			if not adj_s:
				continue
			# Add passable approach tiles surrounding this '#'
			for dx, dy, _ in [(0, -1, "UP"), (0, 1, "DOWN"), (-1, 0, "LEFT"), (1, 0, "RIGHT")]:
				nx, ny = x + dx, y + dy
				if inb(nx, ny) and _is_passable(grid[ny][nx]):
					approach.append((nx, ny))
	return approach


################################################################################
# Tools for overworld node
################################################################################

@tool("map_suggest_path", return_direct=False)
def map_suggest_path(map_display: str, prefer_exit: bool = True, prefer_unknown: bool = True) -> str:
	"""
	Suggest a safe single-step direction (UP/DOWN/LEFT/RIGHT) from 'P' on an ASCII map.
	- map_display: Multiline ASCII map with symbols like P (player), # (wall), D (door), S (stairs), ? (unknown edge), . (floor).
	- prefer_exit: If True, prioritize heading towards exits (D), then stairs (S) if nothing better.
	- prefer_unknown: If True, otherwise head towards '?' frontier tiles to expand exploration.
	- Special heuristic: '#' is not always a hard wall; when a '#' is adjacent to an 'S',
	  treat the adjacent passable tiles as 'frontier approach' targets and prefer moving towards them
	  to explore occluded/new areas instead of bouncing between 'S' tiles.

	Returns a short JSON-like string with keys: next_step, rationale.
	Example:
	{"next_step":"RIGHT","rationale":"Nearest door to the east"}
	"""
	grid, player = _parse_map_display(map_display)
	if not grid or player is None:
		return '{"next_step":"A","rationale":"No valid map or player; proceed interaction."}'
	# 1) Prefer frontier approach tiles near stairs-adjacent '#'
	frontier_approach = _collect_frontier_approach_tiles_near_stairs(grid)
	if frontier_approach:
		# Build a synthetic target mask for BFS: mark approach tiles with a token '@'
		# We won't mutate grid, instead we run BFS checking coordinates membership.
		first_step = _bfs_first_step_to_coordinates(grid, player, set(frontier_approach))
		if first_step:
			return f'{{"next_step":"{first_step}","rationale":"Explore occluded area near stairs (# adjacent to S)"}}'
	# 2) Prefer '?' unknown frontier
	if prefer_unknown:
		first_step = _bfs_first_step_towards_targets(grid, player, ["?"])
		if first_step:
			return f'{{"next_step":"{first_step}","rationale":"Head to unknown frontier ?"}}'
	# 3) Prefer exits (doors) then stairs if enabled
	if prefer_exit:
		first_step = _bfs_first_step_towards_targets(grid, player, ["D"])
		if first_step:
			return f'{{"next_step":"{first_step}","rationale":"Head to exit door (D)"}}'
		first_step = _bfs_first_step_towards_targets(grid, player, ["S"])
		if first_step:
			return f'{{"next_step":"{first_step}","rationale":"Approach stairs (S)"}}'
	# Fallback: pick any passable immediate neighbor
	nbs = _neighbors(player[0], player[1], grid)
	if nbs:
		_, _, dname = nbs[0]
		return f'{{"next_step":"{dname}","rationale":"Fallback neighbor move"}}'
	return '{"next_step":"A","rationale":"No passable moves found"}'


@tool("map_validate_moves", return_direct=False)
def map_validate_moves(map_display: str, moves: List[str]) -> str:
	"""
	Validate a short sequence of moves from the player position on ASCII map.
	- map_display: Multiline ASCII map with 'P' and obstacles '#'
	- moves: List like ["RIGHT","RIGHT","DOWN"]
	Returns a JSON-like string with: valid (bool), fail_index (int or -1), reason (str)
	"""
	grid, player = _parse_map_display(map_display)
	if not grid or player is None:
		return '{"valid":false,"fail_index":0,"reason":"Invalid map or missing player"}'
	x, y = player
	dir_to_delta: Dict[str, Tuple[int, int]] = {
		"UP": (0, -1),
		"DOWN": (0, 1),
		"LEFT": (-1, 0),
		"RIGHT": (1, 0),
	}
	h = len(grid)
	w = len(grid[0]) if h > 0 else 0
	for i, m in enumerate(moves):
		mu = (m or "").upper().strip()
		if mu not in dir_to_delta:
			return f'{{"valid":false,"fail_index":{i},"reason":"Invalid direction {m}"}}'
		dx, dy = dir_to_delta[mu]
		nx, ny = x + dx, y + dy
		if not (0 <= nx < w and 0 <= ny < h):
			return f'{{"valid":false,"fail_index":{i},"reason":"Out of bounds"}}'
		if not _is_passable(grid[ny][nx]):
			return f'{{"valid":false,"fail_index":{i},"reason":"Blocked by wall"}}'
		x, y = nx, ny
	return '{"valid":true,"fail_index":-1,"reason":"All steps passable"}'


@tool("summarize_dialogue_hint", return_direct=False)
def summarize_dialogue_hint(text: str) -> str:
	"""
	Summarize a dialogue line into a compact navigation hint if possible.
	Returns a short sentence like: "Hint: head to Oldale Town" or empty if none.
	"""
	try:
		# Lazy import to avoid circulars
		from .agentic_dialogue_understanding import extract_key_info_from_dialogue
	except Exception:
		# If unavailable, return trimmed text
		t = (text or "").strip()
		return t[:160]
	extraction = extract_key_info_from_dialogue(text or "")
	loc = extraction.get("location")
	dirn = extraction.get("direction")
	if loc and dirn:
		return f"Hint: go {dirn} towards {loc}"
	if loc:
		return f"Hint: head to {loc}"
	if dirn:
		return f"Hint: move {dirn}"
	return ""


# Export a convenience list for the overworld node
OVERWORLD_TOOLS = [
	map_suggest_path,
	map_validate_moves,
	summarize_dialogue_hint,
]

################################################################################
# Additional robustness tools for interaction and anti-stuck behavior
################################################################################

@tool("suggest_interact_alignment", return_direct=False)
def suggest_interact_alignment(map_display: str) -> str:
	"""
	When intending to interact (press 'A') with an object on a wall or adjacent tile,
	suggest a short corrective sequence to face the target and confirm:
	- If a special tile (# as occluded/wall, D door, S stairs) is adjacent, propose [DIR, 'A'] toward it.
	- If multiple candidates exist, pick the closest in reading order: UP, DOWN, LEFT, RIGHT.

	Returns JSON-like: {"actions": ["LEFT","A"], "rationale": "Face wall and interact"}
	"""
	grid, player = _parse_map_display(map_display)
	if not grid or player is None:
		return '{"actions":["A"],"rationale":"Unknown map; proceed with A"}'
	x, y = player
	candidates = []
	for dx, dy, dname in [(0,-1,"UP"), (0,1,"DOWN"), (-1,0,"LEFT"), (1,0,"RIGHT")]:
		nx, ny = x + dx, y + dy
		if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]):
			ch = grid[ny][nx]
			if ch in ("#", "D", "S"):
				candidates.append(dname)
	if candidates:
		first = candidates[0]
		return f'{{"actions":["{first}","A"],"rationale":"Face adjacent target tile and interact"}}'
	# If nothing special adjacent, try approaching '?' frontier first
	first_step = _bfs_first_step_towards_targets(grid, player, ["?"])
	if first_step:
		return f'{{"actions":["{first_step}"],"rationale":"Approach unknown frontier before interacting"}}'
	# Else approach door or stairs
	first_step = _bfs_first_step_towards_targets(grid, player, ["D", "S"])
	if first_step:
		return f'{{"actions":["{first_step}"],"rationale":"Move towards interactable (D/S)"}}'
	return '{"actions":["A"],"rationale":"Default to A; no obvious target"}'


@tool("anti_stuck_unmash", return_direct=False)
def anti_stuck_unmash(context: str, last_actions: list) -> str:
	"""
	Detect repeated 'A' presses outside dialogue and suggest a breaker action.
	- If last 4+ actions are 'A' and not in 'dialogue', suggest 'B' to back out.
	- Otherwise, suggest a perpendicular move to change state.

	Returns JSON-like: {"next_step":"B","rationale":"Stop mashing A"}
	"""
	try:
		acts = [str(a).upper() for a in (last_actions or [])]
	except Exception:
		acts = []
	if context != "dialogue" and len(acts) >= 4 and all(a == "A" for a in acts[-4:]):
		return '{"next_step":"B","rationale":"Prevent A-mash lock; back out"}'
	# Suggest a change of direction to break idle loops
	for candidate in ["RIGHT", "LEFT", "UP", "DOWN"]:
		if not acts or candidate != acts[-1]:
			return f'{{"next_step":"{candidate}","rationale":"Change state to avoid being stuck"}}'
	return '{"next_step":"B","rationale":"Fallback breaker"}'


@tool("break_loop_suggestion", return_direct=False)
def break_loop_suggestion(last_actions: list) -> str:
	"""
	Detect back-and-forth loops like LEFT/RIGHT or UP/DOWN and suggest a perpendicular move.
	Returns JSON-like: {"next_step":"UP","rationale":"Break LR loop"}
	"""
	try:
		acts = [str(a).upper() for a in (last_actions or [])]
	except Exception:
		acts = []
	if len(acts) < 4:
		return '{"next_step":"A","rationale":"No strong loop pattern; proceed"}'
	p = "".join(acts[-4:])
	if all(a in ("LEFT","RIGHT") for a in acts[-4:]):
		return '{"next_step":"UP","rationale":"Break horizontal ping-pong"}'
	if all(a in ("UP","DOWN") for a in acts[-4:]):
		return '{"next_step":"RIGHT","rationale":"Break vertical ping-pong"}'
	return '{"next_step":"A","rationale":"No loop detected"}'


# Extend exported tool list
OVERWORLD_TOOLS.extend([
	suggest_interact_alignment,
	anti_stuck_unmash,
	break_loop_suggestion,
])

################################################################################
# Memory tools (allow the model to explicitly manage long-term objectives/focus)
################################################################################
from .agentic_longterm_memory import get_global_memory

@tool("memory_status", return_direct=False)
def memory_status() -> str:
	"""
	Return a concise summary of the agent's long-term memory:
	- Focus, active objectives, recently completed, dialogue hints, top visited.
	"""
	mem = get_global_memory()
	try:
		return mem.build_summary()
	except Exception:
		return "No prior memory."


@tool("memory_add_objective", return_direct=False)
def memory_add_objective(objective_type: str, description: str, target_value: str = "") -> str:
	"""
	Add a new high-level objective to long-term memory.
	Returns the created objective id.
	"""
	mem = get_global_memory()
	oid = mem.add_objective(description=description, objective_type=objective_type, target_value=target_value or None)
	mem.save()
	return f"ADDED:{oid}"


@tool("memory_complete_objective", return_direct=False)
def memory_complete_objective(objective_id: str, notes: str = "") -> str:
	"""
	Mark an objective as completed in long-term memory.
	"""
	mem = get_global_memory()
	ok = mem.complete_objective(objective_id, notes)
	mem.save()
	return "COMPLETED" if ok else "NOT_FOUND"


@tool("memory_remember_focus", return_direct=False)
def memory_remember_focus(text: str) -> str:
	"""
	Update the memory's last_focus string so subsequent steps keep consistent direction.
	"""
	mem = get_global_memory()
	mem.set_last_focus(text or "")
	mem.save()
	return "FOCUS_UPDATED"

@tool("memory_npc_add_task", return_direct=False)
def memory_npc_add_task(key: str, description: str, source: str = "") -> str:
	"""
	Add or update an NPC-assigned task in long-term memory.
	Returns OK.
	"""
	mem = get_global_memory()
	mem.npc_upsert_task(key or "", description or "", source_npc=source or "", notes="tool_add")
	mem.save()
	return "OK"

@tool("memory_npc_complete_task", return_direct=False)
def memory_npc_complete_task(key: str, notes: str = "") -> str:
	"""
	Mark an NPC task as completed.
	Returns COMPLETED or NOT_FOUND.
	"""
	mem = get_global_memory()
	ok = mem.npc_mark_completed(key or "", notes or "tool_complete")
	mem.save()
	return "COMPLETED" if ok else "NOT_FOUND"

@tool("memory_npc_status", return_direct=False)
def memory_npc_status() -> str:
	"""
	Return concise status of NPC tasks: pending and done keys.
	"""
	mem = get_global_memory()
	try:
		pending = [k for k, v in mem.npc_tasks.items() if not v.get("completed")]
		done = [k for k, v in mem.npc_tasks.items() if v.get("completed")]
		return f"PENDING:{pending} | DONE:{done}"
	except Exception:
		return "PENDING:[] | DONE:[]"

# Expose memory tools to overworld as well
OVERWORLD_TOOLS.extend([
	memory_status,
	memory_add_objective,
	memory_complete_objective,
	memory_remember_focus,
	memory_npc_add_task,
	memory_npc_complete_task,
	memory_npc_status,
])

################################################################################
# Dialogue tools
################################################################################

@tool("dialogue_choice_helper", return_direct=False)
def dialogue_choice_helper(prefer: str = "YES", current_selection: str = "") -> str:
	"""
	Provide a safe generic sequence to select a YES/NO choice.
	- If prefer == "YES": return ["UP","A"] (most Pokémon UIs place YES above NO)
	- If prefer == "NO": return ["A"] (default cursor often starts on NO; else use DOWN, A)
	- If current_selection is known ("YES" or "NO"), return minimal move then A

	Returns JSON-like: {"actions":["UP","A"],"rationale":"Select YES safely"}
	"""
	p = (prefer or "YES").strip().upper()
	cs = (current_selection or "").strip().upper()
	if p == "YES":
		if cs == "YES":
			return '{"actions":["A"],"rationale":"Already on YES; confirm"}'
		return '{"actions":["UP","A"],"rationale":"Move to YES then confirm"}'
	if p == "NO":
		if cs == "NO":
			return '{"actions":["A"],"rationale":"Already on NO; confirm"}'
		# Unknown selection; DOWN then A is safer than assuming default
		return '{"actions":["DOWN","A"],"rationale":"Move to NO then confirm"}'
	# Unknown preference: default to confirm with A
	return '{"actions":["A"],"rationale":"Default confirm"}'

DIALOGUE_TOOLS = [
	anti_stuck_unmash,
	dialogue_choice_helper,
]

# Make dialogue choice helper available in overworld as well (dialogue can appear there)
OVERWORLD_TOOLS.extend([
	dialogue_choice_helper,
])

@tool("task_aware_path", return_direct=False)
def task_aware_path(map_display: str, pending_tasks: list = None) -> str:
	"""
	Task-aware path suggestion: when there are pending NPC tasks in the current area,
	avoid stepping onto exits like 'S' (stairs) or 'D' (door).
	Strategy:
	1) Approach '#' adjacent to 'S' (occluded near stairs) via safe approach tiles.
	2) Move toward '?' frontier.
	3) As a fallback, choose any immediate passable neighbor that is NOT 'S' or 'D'.
	Returns JSON-like: {"next_step":"RIGHT","rationale":"Avoid exits while tasks pending"}
	"""
	grid, player = _parse_map_display(map_display)
	if not grid or player is None:
		return '{"next_step":"A","rationale":"Invalid map; proceed with A"}'
	# If no pending tasks provided, treat as normal suggest_path without exits
	pending = pending_tasks or []
	# Prefer frontier approach near stairs
	approach = _collect_frontier_approach_tiles_near_stairs(grid)
	if approach and pending:
		first = _bfs_first_step_to_coordinates(grid, player, set(approach))
		if first:
			return f'{{"next_step":"{first}","rationale":"Explore near stairs without exiting"}}'
	# Prefer '?'
	first = _bfs_first_step_towards_targets(grid, player, ["?"])
	if first:
		return f'{{"next_step":"{first}","rationale":"Head to frontier while avoiding exits"}}'
	# Fallback: pick any neighbor that is passable and not an exit
	nbs = _neighbors(player[0], player[1], grid)
	for nx, ny, dname in nbs:
		ch = grid[ny][nx]
		if ch not in ("S", "D"):
			return f'{{"next_step":"{dname}","rationale":"Safe neighbor avoiding exits"}}'
	# Last resort: any neighbor
	if nbs:
		_, _, dname = nbs[0]
		return f'{{"next_step":"{dname}","rationale":"No safe alternative; neighbor move"}}'
	return '{"next_step":"A","rationale":"No moves"}'

# Add task-aware path to overworld tools
OVERWORLD_TOOLS.extend([
	task_aware_path
])