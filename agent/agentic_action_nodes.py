'''
4 Different Action Nodes for 4 game contexts: 
1) overall world 
2) battle 
3) dialogue 
4) menu 
'''

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import BaseModel
from .agentic_utils import _to_data_url
from .agentic_system_prompts import overworld_system_prompt, unknow_system_prompt, battle_system_prompt, dialogue_system_prompt, menu_system_prompt


def action_node(agent_state, action_system_msg, tools_available, vlm, OutputTemplate, name, system_prompt=None):
    '''
    general action nodes can be used by any functions, including overworld, battle, dialogue and menu others 
    '''
    print('========IN_ACTION_NODE========\n')
    # Choose system prompt based on mode if not explicitly provided
    system_msg = SystemMessage(content=(action_system_msg))

     # Build textual context for the model
    last_5_actions = agent_state.get("last_5_actions", "")
    last_action_reason = agent_state.get("last_action_reason", "")
    formatted_state = agent_state.get("formatted_game_state", "")
    current_plan = agent_state.get("current_plan","")
    map_name =agent_state.get("map_name","")
    important_info_from_dialogue = agent_state.get("important_info_from_dialogue", '')
    # map_memory = agent_state.get("map_memory", "")

    text_msg = (
        f"Current Focus: {current_plan}\n"
        f"Last 5 actions: {last_5_actions}\n"
        f"Last action reasoning: {last_action_reason}\n"
        f"Important information from previous dialogue: {important_info_from_dialogue} \n"
        f"Current in Map {map_name} with following game state:\n{formatted_state}"
        # f"You have been to these position: {map_memory}"
    )
    text_input = [{"type": "text", "text": text_msg}]

    # Accept pre-encoded data URL or raw image in agent_state
    image_url = agent_state.get("image_url")
    image_obj = agent_state.get("image")  # PIL or numpy
    if not image_url and image_obj is not None:
        image_url = _to_data_url(image_obj)
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
    print('========CHECK_RES_OF_ACTION_NODE========\n')
    print(parsed)

    return {
        "actions": parsed.action,
        "reasons": parsed.reason,
         "messages": [raw_result],
        "sender": name,
        "important_info_from_dialogue":parsed.important_info_from_dialogue}