'''
planner nodes with functions: 
1) coordinate the proper action notes to work; 
2) plan the long term objectives; 
3) summarize the last_5_actions_reasoning. 
'''
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel
from .agentic_utils import _to_data_url

def planner_node(agent_state, tools_available, vlm, OutputTemplate, name, system_prompt = None):
    
    print('========IN_PLANNER_NODE========\n')
    system_msg = SystemMessage(content=(system_prompt))

    # mode = agent_state.get("mode", "overworld")
    map_name =agent_state.get("map_name","")
    previous_plan = agent_state.get("previous_plan", "")
    # last_5_actions = agent_state.get("last_5_actions", "")
    formatted_state = agent_state.get("formatted_game_state", "")

    text_msg = (
        f"Player's Current Location: {map_name}\n"
        f"Previous Plan: {previous_plan}\n"
        f"Current Game State: {formatted_state}"
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

    def _msgs(_):
        return [system_msg, text_input_msg]

    chain = RunnableLambda(_msgs) | vlm.bind_tools(tools_available).with_structured_output(OutputTemplate)
    parsed = chain.invoke({})
    print('========END_OF_PLANNER_NODE========\n')
    return {
        "current_analysis": parsed.analysis,
        "current_plan": parsed.plan,
        "sender": name
    }


    