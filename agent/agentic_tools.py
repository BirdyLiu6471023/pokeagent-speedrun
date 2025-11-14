from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, AIMessage 
import random
import logging
from utils.vlm import VLM
from utils.state_formatter import format_state_for_llm, format_state_summary, get_movement_options, get_party_health_summary
from agent.system_prompt import system_prompt
# Set up module logging
logger = logging.getLogger(__name__)

