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
from langchain.output_parsers import PydanticOutputParser

## import libraries related to graph 
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

# Import LLM logger
from utils.llm_logger import log_llm_interaction, log_llm_error

# # import project functions
# from utils.state_formatter import format_state_for_llm, format_state_summary, get_movement_options, get_party_health_summary
# from agent.system_prompt import system_prompt

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


def agent_node(agent_state, tools_available, vlm, OutputTemplate, name):
    # Build multimodal messages (text + optional image)
    system_msg = SystemMessage(content=(
        "You are a player who is playing Pokemon Emerald"
        f" You have tools including {tools_available}. "
    ))

    # Accept pre-encoded data URL or raw image in agent_state
    image_url = agent_state.get("image_url")
    image_obj = agent_state.get("image")  # PIL or numpy
    if not image_url and image_obj is not None:
        image_url = _to_data_url(image_obj)

    human_input = [{"type": "text", "text": agent_state['text_msg']}]  # required text
    if image_url:
        human_input.append({"type": "image_url", "image_url": {"url": image_url}})

    human_msg = HumanMessage(content=human_input)

    # First, run a normal tool-callable message to preserve tool routing info
    raw_agent = vlm.bind_tools(tools_available)
    raw_result = raw_agent.invoke([system_msg, human_msg])

    # Then, use LangChain's pipe style with structured output
    def _msgs(_):
        return [system_msg, human_msg]

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
    text_msg: str
    image: Union[Image.Image, np.ndarray]
    actions: Annotated[List[BaseMessage], operator.add]
    reasons: Annotated[List[BaseMessage], operator.add]


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

        # Tools
        self.calculation_tools = [multiplyer, add]

        # Model
        self.vlm = ChatOpenAI(model_name='gpt-4o')

        # Output schema
        class StructuredOutput(BaseModel):
            action: Literal['A', 'B', 'SELECT', 'UP', 'DOWN', 'RIGHT', 'LEFT']
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
        enter_chain = {"text_msg": self.default_prompt, "image": frame}
        start_time = time.time()
        res = self._chain.invoke(enter_chain)
        duration = time.time() - start_time
        action_decide = res['actions'][-1]
        reason_for_action = res['reasons'][-1]
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



if __name__ == "__main__":
    from PIL import Image
    img = Image.open("emerald.png")  # or a numpy array frame
    result = action_chain("Please look at the game part of the picture, what action should the play to do?", frame=img)
    print(result)
