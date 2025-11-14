'''
Runnable Framework using Langgraph and Langchain framework. 
'''

import operator
from typing import TypedDict, Annotated, List, Literal, Union, Optional, Any
from dataclasses import field
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import BaseModel

## import libraries related to graph 
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

# Import LLM logger
from utils.llm_logger import log_llm_interaction, log_llm_error
from utils.state_formatter import format_state_for_llm

from .agentic_utils import get_game_context, get_map_id, get_player_coords, get_not_suggested_directions, get_map_name, get_current_map_display, _decode_base64_image, extract_frame_from_game_state,  _to_data_url
from .agentic_system_prompts import overworld_system_prompt, planning_system_prompt, unknow_system_prompt,  battle_system_prompt, dialogue_system_prompt, menu_system_prompt
from .agentic_action_nodes import action_node 
from .agentic_planner_node import planner_node 

# Set up module logging
import logging
logger = logging.getLogger(__name__)

import os
import time
from dotenv import load_dotenv, find_dotenv
from io import BytesIO
from PIL import Image
import base64
import numpy as np
import functools
load_dotenv(find_dotenv())

print("libraries installed successfully") 

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class GraphState(TypedDict):
    ## current action message: 
    messages:Annotated[List[BaseMessage], operator.add]

    ## Graph Routing Usage:  
    sender:str 

    ## memory or history: 
    # last actions, previous plan
    last_5_actions: str
    last_action_reason: str 
    previous_plan: str

    ## current obs: 
    mode: str 
    formatted_game_state: str
    image: Union[Image.Image, np.ndarray]
        ## map related states, map_name, pos, map_display, map movement history 
    map_name: Optional[str]
    pos: Optional[tuple]
    map_display: Optional[str]
    visit_memory: dict

    ## agent status: 
    ## stuck: in map, the agent stuck in 5 steps. 
        # <DEVELOPMENT>

    ## current_planning 
    current_plan: str 
    current_analysis:str
    important_info_from_dialogue: Optional[str]

    # action output: 
    actions: List[str]
    reasons: str

Buttons = Literal['A', 'B', 'SELECT', 'UP', 'DOWN', 'RIGHT', 'LEFT', 'L', 'R']

class ActionStructuredOutput(BaseModel):
    action: List[Buttons]
    reason: str
    important_info_from_dialogue: Optional[str]

class PlannerStructuredOutput(BaseModel):
    analysis: str
    plan:str

# class Objective(BaseModel):
#     """Single objective/goal for the agent"""
#     id: str
#     description: str
#     objective_type: str  # "location", "battle", "item", "dialogue", "custom"
#     target_value: Optional[Any] = None  # Specific target (coords, trainer name, item name, etc.)
#     completed: bool = False
#     created_at: datetime = field(default_factory=datetime.now)
#     completed_at: Optional[datetime] = None
#     progress_notes: str = ""
#     storyline: bool = False  # True for main storyline objectives (auto-verified), False for agent sub-objectives
#     milestone_id: Optional[str] = None  # Emulator milestone ID for storyline objectives


class OpenAIAgenticFramework:
    """Callable agentic framework using OpenAI gpt that returns (action, reason)."""

    def __init__(self, default_prompt: Optional[str] = None, model_name = 'gpt-4o'):
        self.default_prompt = default_prompt or (
            "Please look at the game frame and choose the best next button action."
        )
        # vlm Model
        self.planner_vlm = ChatOpenAI(model_name='gpt-5') 
        self.action_vlm = ChatOpenAI(model_name = 'gpt-4o')
        self.step_count = 0
        # Rolling Histories: 
        self._chain_res_history: list[dict] = []
        self._action_history: list[list] = []
        self._visit_memory: dict = {}
        self._reason_history: list[str] = []
        self._plan_history: list[str] = []

        # Tools
        self.overworld_tools = []
        self.overworld_tool_node = ToolNode(self.overworld_tools) 

        self.battle_tools = []
        self.battle_tool_node = ToolNode(self.battle_tools)

        # self.menu_tools = []
        # self.menu_tool_node = ToolNode(self.menu_tools)

        # self.dialogue_tools = []
        # self.dialogue_tool_node = ToolNode(self.dialogue_tools)

        # self.planner_tools = []
        # self.planner_tool_node = ToolNode(self.planner_tools)

        self.ActionOutputTemplate = ActionStructuredOutput
        self.PlannerOutputTemplate = PlannerStructuredOutput

        # Action Nodes 
        self.overworld_action_node = functools.partial(
            action_node,
            action_system_msg = overworld_system_prompt,
            tools_available=self.overworld_tools, 
            vlm=self.action_vlm, 
            OutputTemplate= self.ActionOutputTemplate,
            name='action_overworld' 
        )

        self.battle_action_node = functools.partial(
            action_node, 
            action_system_msg = battle_system_prompt,
            tools_available = self.battle_tools, 
            vlm = self.action_vlm, 
            OutputTemplate = self.ActionOutputTemplate,
            name = 'action_battle'
        )

        self.other_action_node = functools.partial(
            action_node, 
            action_system_msg = unknow_system_prompt, 
            tools_available = [],
            vlm = self.action_vlm, 
            OutputTemplate = self.ActionOutputTemplate,
            name = "action_other"
        )

        ## think of what long term objective to focus? 
        self.planner_node = functools.partial(
            planner_node,
            tools_available =[],
            vlm =self.planner_vlm, 
            OutputTemplate = self.PlannerOutputTemplate,
            name = 'planner',
            system_prompt = planning_system_prompt
        )

        # ## think of current game state
        # self.perception_node = ()

        # ## memory 
        # self.memory_node = ()

        ## planner_graph 
        planner_graph = StateGraph(GraphState) 

        def _stater_router(state):
            if state['step_count'] % 10 == 0:
                return 'planner'
            else:
                mode = state['mode']
                if mode == 'overworld':
                    return "overworld"
                elif mode == 'battle':
                    return 'battle'
                else:
                    return "other"


        # routing mechanism 
        def _router(state):
            messages = state['messages']
            last_message = messages[-1]
            return "continue" if not getattr(last_message, 'tool_calls', None) else "call_tool"
        
        def _planner_router(state):
            mode = state['mode']
            if mode == 'overworld':
                return "overworld"
            elif mode == 'battle':
                return 'battle'
            else:
                return "other"

        planner_graph.add_node("overworld_action_node", self.overworld_action_node)
        planner_graph.add_node("overworld_tool_node", self.overworld_tool_node)
        planner_graph.add_node("battle_action_node", self.battle_action_node)
        planner_graph.add_node("battle_tool_node", self.battle_tool_node)
        planner_graph.add_node("other_action_node", self.other_action_node)
        planner_graph.add_node("planner_node", self.planner_node)
        # graph.add_node("perception_node", self.perception_node)
        # graph.add_node("memory_node", self.memory_node)

        planner_graph.add_conditional_edges(START, _stater_router, {"planner":"planner_node","overworld": "overworld_action_node", "other": "other_action_node", 'battle':'battle_action_node'})
        planner_graph.add_conditional_edges("planner_node", _planner_router, {"overworld": "overworld_action_node", "other": "other_action_node", 'battle':'battle_action_node'})
        
        planner_graph.add_conditional_edges("overworld_action_node", _router, {"call_tool": "overworld_tool_node", "continue": END})
        planner_graph.add_edge("overworld_tool_node", "overworld_action_node")

        planner_graph.add_conditional_edges("battle_action_node", _router, {"call_tool": "battle_tool_node", "continue": END})
        planner_graph.add_edge("battle_tool_node", "battle_action_node")

        planner_graph.add_edge("other_action_node", END)

        self.planner_chain = planner_graph.compile()

        # Optional, zero-impact graph export toggled by env var
        if os.getenv("POKEAGENT_PLOT_GRAPH") == "1":
            try:
                output_path = os.getenv("POKEAGENT_PLOT_GRAPH_PATH", "planner_graph.png")
                png_bytes = self.planner_chain.get_graph().draw_mermaid_png()
                with open(output_path, "wb") as f:
                    f.write(png_bytes)
                logger.info("Planner graph PNG exported to %s", output_path)
            except Exception:
                logger.exception("Failed to export planner graph PNG")

        ## action_graph 


    def __call__(self, game_state: dict) -> tuple[list[str], str]:
        frame = extract_frame_from_game_state(game_state)
        context_mode = get_game_context(game_state)
        formatted_state = format_state_for_llm(game_state)
        map_name = get_map_name(game_state)
        pos = get_player_coords(game_state)
        step_count = self.step_count
        map_display = get_current_map_display(game_state)
        print("====CONTEXT_MODE======\n", context_mode)
        print("=======MAP_DISPLAY==== \n", map_display)

        # Prepare recent history summaries
        if self._action_history:
            last5_actions = " | ".join("".join(step_actions) for step_actions in self._action_history[-5:])
        else:
            last5_actions = ""
        # Initialize per-map memory safely and compute suggestions if we have required data
        
        if context_mode == 'overworld' and map_name is not None and pos is not None:
            if map_name not in self._map_memory:
                self._map_memory[map_name] = {}
            
            if pos not in self._map_memory[map_name]: 
                self._map_memory[map_name][pos] = 1 
            else:
                self._map_memory[map_name][pos] += 1

        # Keep reasons concise for prompt context
        last_action_reason = self._reason_history[-1] if self._reason_history else ""
        
        previous_plan = self._plan_history[-1] if self._plan_history else ''

        

        enter_chain = {
            "mode": context_mode,
            "image": frame,
            "formatted_game_state": formatted_state,

            "last_5_actions": last5_actions,
            "last_action_reason": last_action_reason,
            
            "map_name":map_name,
            "pos": pos,
            "map_display":map_display,

            "current_plan":previous_plan,
            "previous_plan":previous_plan,
            "step_count": self.step_count, 
            # 'map_memory':self._map_memory
        }

        start_time = time.time()

        res = self.planner_chain.invoke(enter_chain)
        
        # if self.step_count == 0:
        #     Image(self.planner_chain.get_graph().draw_mermaid_png())

        duration = time.time() - start_time

        action_decide = res['actions']
        reason_for_action = res['reasons']
        self.step_count += 1
        
        # Update histories (cap to a reasonable size)
        self._action_history.append(action_decide)
        self._reason_history.append(reason_for_action)
        if len(self._action_history) > 100:
            self._action_history = self._action_history[-100:]
        if len(self._reason_history) > 100:
            self._reason_history = self._reason_history[-100:]


        try:
            # Log to LLM logger so /stream can display reasoning per step
            log_llm_interaction(
                interaction_type="agentic_framework_action",
                prompt=self.default_prompt,
                response=f"ACTION: {action_decide}\nREASONING: {reason_for_action}",
                metadata={},
                duration=duration,
                model_info={"model": "openai/gpt-5"}
            )
        except Exception:
            pass
        return action_decide, reason_for_action


def get_agentic_framework(name: Optional[str] = None):
    """Return a callable framework based on name. Defaults to OpenAI gpt-4o.

    Usage:
        self.agent_framework = get_agentic_framework()  # defaults to OpenAI gpt-4o
        action, reason = self.agent_framework(game_state)
    """
    key = (name or "openai").lower()
    if key in ("openai", "gpt-4o", "openai:gpt-5"):
        return OpenAIAgenticFramework()
    # Future: add other frameworks here
    return OpenAIAgenticFramework()

