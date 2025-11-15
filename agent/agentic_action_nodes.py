'''
4 Different Action Nodes for 4 game contexts: 
1) overall world 
2) battle 
3) dialogue 
4) other 
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
    memory_summary = agent_state.get("memory_summary","")
    last_actions_list = agent_state.get("last_actions_list", [])
    pending_npc_tasks = agent_state.get("pending_npc_tasks", [])
    # Enumerate tool names for the model to discover and call them
    try:
        tool_names = [getattr(t, "name", None) or getattr(t, "__name__", "tool") for t in (tools_available or [])]
        tool_names_line = ", ".join([str(n) for n in tool_names if n])
    except Exception:
        tool_names_line = ""

    text_msg = (
        f"Current Focus: {current_plan}\n"
        f"Last 5 actions: {last_5_actions}\n"
        f"Recent actions list (for tools): {last_actions_list}\n"
        f"Last action reasoning: {last_action_reason}\n"
        f"Pending NPC tasks: {pending_npc_tasks}\n"
        f"Important information from previous dialogue: {important_info_from_dialogue} \n"
        f"Memory: {memory_summary}\n"
        f"Current in Map {map_name} with following game state:\n{formatted_state}\n"
        f"Tools available: {tool_names_line}\n"
        f"- Use 'map_suggest_path' for single-step pathing to frontier/exits.\n"
        f"- Use 'suggest_interact_alignment' to face adjacent targets and interact.\n"
        f"- Use 'anti_stuck_unmash' to stop A-mashing outside dialogue.\n"
        f"- Use 'break_loop_suggestion' to break LR/UD loops.\n"
        f"- Use 'map_validate_moves' to verify short move chains.\n"
        f"- If there are pending NPC tasks, avoid exiting via 'S' or 'D'; complete tasks first.\n"
        f"- For YES/NO menus, use 'dialogue_choice_helper' (prefer='YES') then confirm.\n"
        f"- In YES/NO menu, use 'dialogue_choice_helper' (prefer='YES') then confirm."
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
    # Tool usage introspection
    used_tools = []
    try:
        tool_calls = getattr(raw_result, 'tool_calls', None)
        if tool_calls:
            for tc in tool_calls:
                try:
                    name_ = tc.get('name') if isinstance(tc, dict) else getattr(tc, 'name', None)
                    if name_:
                        used_tools.append(str(name_))
                except Exception:
                    continue
    except Exception:
        pass
    if used_tools:
        print(f"TOOLS CALLED: {', '.join(used_tools)}")

    # Ensure important_info_from_dialogue is populated (fallback to agent state's hint)
    info_from_dialogue = getattr(parsed, "important_info_from_dialogue", None)
    if not info_from_dialogue:
        info_from_dialogue = agent_state.get("important_info_from_dialogue", "") or ""

    # Append tool usage info to reasons for transparency
    reasons_out = parsed.reason
    if used_tools:
        reasons_out = f"{parsed.reason} | Used tools: {', '.join(used_tools)}"

    return {
        "actions": parsed.action,
        "reasons": reasons_out,
        "messages": [raw_result],
        "sender": name,
        "important_info_from_dialogue": info_from_dialogue,
        "tool_loop_count": (agent_state.get("tool_loop_count", 0) + (1 if used_tools else 0))}