'''
Runnable Framework using Langgraph and Langchain. 
'''


import operator
from tkinter import END
from typing import TypedDict, Annotated, List, Literal, Union, Optional
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
# # import project functions
# from utils.state_formatter import format_state_for_llm, format_state_summary, get_movement_options, get_party_health_summary
# from agent.system_prompt import system_prompt

from .agentic_utils import get_game_context 
from .agentic_system_prompts import overworld_system_prompt, planning_system_prompt, unknow_system_prompt

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

# ------------------------
# Frame utilities
# ------------------------
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

@tool
def multiplyer(a, b):
    'this is a function to multiply a and b.'
    return a*b 

@tool
def add(a, b):
    'this is a function to add a and b'
    return a+b


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


def agent_node(agent_state, tools_available, vlm, OutputTemplate, name, system_prompt=None):
    # Choose system prompt based on mode if not explicitly provided
    mode = agent_state.get("mode", "overworld")
    if not system_prompt:
        if mode == "overworld":
            system_prompt = overworld_system_prompt
        else:
            system_prompt = unknow_system_prompt
    system_msg = SystemMessage(content=(system_prompt))

    # Accept pre-encoded data URL or raw image in agent_state
    image_url = agent_state.get("image_url")
    image_obj = agent_state.get("image")  # PIL or numpy
    if not image_url and image_obj is not None:
        image_url = _to_data_url(image_obj)

    # Build textual context for the model
    last_5_actions = agent_state.get("last_5_actions", "")
    last_5_reasoning_summary = agent_state.get("last_5_reasoning_summary", "")
    formatted_state = agent_state.get("formatted_game_state", "")
    text_msg = (
        f"Mode: {mode}\n"
        f"Last 5 actions: {last_5_actions}\n"
        f"Recent reasoning: {last_5_reasoning_summary}\n\n"
        f"State:\n{formatted_state}"
    )
    text_input = [{"type": "text", "text": text_msg}]

    if image_url:
        text_input.append({"type": "image_url", "image_url": {"url": image_url}})

    text_input_msg = HumanMessage(content=text_input)

    # First, run a normal tool-callable message to preserve tool routing info
    raw_agent = vlm.bind_tools(tools_available)
    raw_result = raw_agent.invoke([system_msg, text_input_msg])

    # Then, use LangChain's pipe style with structured output
    def _msgs(_):
        return [system_msg, text_input_msg]

    chain = RunnableLambda(_msgs) | vlm.bind_tools(tools_available).with_structured_output(OutputTemplate)
    parsed = chain.invoke({})

    return {
        "actions": [parsed.action],
        "reasons": [parsed.reason],
        "messages": [raw_result],
        "sender": name
    }


class ActionState(TypedDict):
    messages:Annotated[List[BaseMessage], operator.add]
    sender:str
    image: Union[Image.Image, np.ndarray]
    actions: Annotated[List[BaseMessage], operator.add]
    reasons: Annotated[List[BaseMessage], operator.add]
    last_5_actions: str
    last_5_reasoning_summary: str 
    mode: str 
    formatted_game_state: str
    


# def action_chain(human_input, frame=None):
    
#     openai_vlm = ChatOpenAI(model_name = 'gpt-4o') 

#     calculation_tools = [multiplyer, add]

#     calculation_tools_node = ToolNode(calculation_tools)

#     class StructuredOutput(BaseModel):
#         action: Literal['A', 'B', 'SELECT', 'UP', 'DOWN', 'RIGHT', 'LEFT']
#         reason: str

#     action_node = functools.partial(agent_node, tools_available = calculation_tools, vlm = openai_vlm, OutputTemplate = StructuredOutput,name = 'action')

#     def router(state):
#         messages = state['messages']
#         last_message = messages[-1]
#         ## ? not inistance last message? 
#         # if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
#         if not last_message.tool_calls:
#             return "continue"
#         else:
#             return "call_tool"
    
#     graph = StateGraph(ActionState)

#     graph.add_node("action_maker", action_node)
#     graph.add_node("calculation_tools", calculation_tools_node)
#     graph.add_edge(START, "action_maker")
#     graph.add_conditional_edges("action_maker", 
#     router, 
#     {"call_tool":"calculation_tools", "continue": END}
#     )

#     graph.add_edge("calculation_tools", END)

#     chain = graph.compile()

#     # If a frame is provided (PIL or numpy), it will be converted to data URL in agent_node
#     enter_chain = {"text_msg": human_input, "image": frame}

#     res = chain.invoke(enter_chain)

#     action_decide = res['actions'][-1]
#     reason_for_action = res['reasons'][-1]

#     return action_decide, reason_for_action


class OpenAIAgenticFramework:
    """Callable agentic framework using OpenAI gpt-4o that returns (action, reason)."""

    def __init__(self, default_prompt: Optional[str] = None):
        self.default_prompt = default_prompt or (
            "Please look at the game frame and choose the best next button action."
        )

        # Keep short rolling histories so we can provide recent context to the chain
        self._action_history: list[str] = []
        self._reason_history: list[str] = []

        # Tools
        self.calculation_tools = [multiplyer, add]

        # Model
        self.vlm = ChatOpenAI(model_name='gpt-4o')

        # Output schema
        class StructuredOutput(BaseModel):
            action: Literal['A', 'B', 'SELECT', 'UP', 'DOWN', 'RIGHT', 'LEFT', 'L', 'R']
            reason: str
        self.OutputTemplate = StructuredOutput

        # LangGraph
        self._action_node = functools.partial(
            agent_node,
            tools_available=self.calculation_tools,
            vlm=self.vlm,
            OutputTemplate=self.OutputTemplate,
            name='action'
        )
        self._tool_node = ToolNode(self.calculation_tools)

        graph = StateGraph(ActionState)
        graph.add_node("action_maker", self._action_node)
        graph.add_node("calculation_tools", self._tool_node)
        graph.add_edge(START, "action_maker")

        def _router(state):
            messages = state['messages']
            last_message = messages[-1]
            return "continue" if not getattr(last_message, 'tool_calls', None) else "call_tool"

        graph.add_conditional_edges("action_maker", _router, {"call_tool": "calculation_tools", "continue": END})
        graph.add_edge("calculation_tools", END)
        self._chain = graph.compile()

    def __call__(self, game_state: dict) -> tuple[str, str]:
        frame = extract_frame_from_game_state(game_state)
        context_mode = get_game_context(game_state)
        formatted_state = format_state_for_llm(game_state)
        # Prepare recent history summaries
        last5_actions = ", ".join(self._action_history[-5:]) if self._action_history else ""
        # Keep reasons concise for prompt context
        if self._reason_history:
            last5_reasons_list = self._reason_history[-5:]
            # Trim each reason to avoid overly long prompts
            trimmed = [r if len(r) <= 200 else (r[:197] + "...") for r in last5_reasons_list]
            last5_reasons = " | ".join(trimmed)
        else:
            last5_reasons = ""

        enter_chain = {
            "mode": context_mode,
            "image": frame,
            "formatted_game_state": formatted_state,
            "last_5_actions": last5_actions,
            "last_5_reasoning_summary": last5_reasons,
        }
        start_time = time.time()
        res = self._chain.invoke(enter_chain)
        print(res) 
        duration = time.time() - start_time
        action_decide = res['actions'][-1]
        reason_for_action = res['reasons'][-1]
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
                model_info={"model": "openai/gpt-4o"}
            )
        except Exception:
            pass
        return action_decide, reason_for_action

    # # Convenience alias for code that expects a .step(...) method
    # def step(self, game_state: dict) -> tuple[str, str]:
    #     return self.__call__(game_state)


def get_agentic_framework(name: Optional[str] = None):
    """Return a callable framework based on name. Defaults to OpenAI gpt-4o.

    Usage:
        self.agent_framework = get_agentic_framework()  # defaults to OpenAI gpt-4o
        action, reason = self.agent_framework(game_state)
    """
    key = (name or "openai").lower()
    if key in ("openai", "gpt-4o", "openai:gpt-4o"):
        return OpenAIAgenticFramework()
    # Future: add other frameworks here
    return OpenAIAgenticFramework()



# if __name__ == "__main__":
#     from PIL import Image
#     img = Image.open("emerald.png")  # or a numpy array frame
#     result = action_chain("Please look at the game part of the picture, what action should the play to do?", frame=img)
#     print(result)
