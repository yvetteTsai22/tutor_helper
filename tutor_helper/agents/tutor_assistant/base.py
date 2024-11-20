"""Agent that will be responsible to define KB structure and orchestrate tasks to KB workers."""
from typing import Any, Dict, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent import AgentExecutor
from langchain.callbacks.manager import Callbacks
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent

from langchain.base_language import BaseLanguageModel
from langchain.agents.agent_toolkits.base import BaseToolkit

from tutor_helper.common.llms import LlmLoader
from tutor_helper.callbacks.stdout_all import StdOutAllCallbackHandler
from tutor_helper.agents.tutor_assistant.toolkit import (
    SimplifiedToolkit,
)

# from trend_langchain_baha.agents.tech_support_copilot.prompt import (
from tutor_helper.agents.tutor_assistant.prompt_chat_agent import (
    PREFIX,
    SUFFIX,
    FORMAT_INSTRUCTIONS,
    INPUT_VARIABLES,
)

from tutor_helper.output_parsers.agent_parser import NewAgentOutputFixingParser

# from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.prompts import MessagesPlaceholder

from langchain.memory import ConversationBufferMemory
# from tutor_helper.memory.buffer import ConversationBufferMemory

import uuid


def chat_agent(
    llm: BaseLanguageModel = None,  # <- Language model to use, will be initialized if not provided
    toolkit: BaseToolkit = None,  # <- Toolkit for KB Drafter, will be loading default if not provided
    format_instructions: str = FORMAT_INSTRUCTIONS,  # <- Default format instructions for the comms with LLM (langchain)
    verbose: bool = False,
    memory: Optional[ConversationBufferMemory] = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    ),
    callbacks: Callbacks = [StdOutAllCallbackHandler()],
    llm_kwargs: Optional[dict] = None,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    agent_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:

    # Loading default llm if not provided
    llm = llm or LlmLoader.create_chat_llm(
        # model = LlmLoader.DEPLOYMENT_GPT4_STD,
        callbacks=callbacks,
        **(llm_kwargs or {}),
    )

    # Loading default toolkit if not provided
    toolkit_set = toolkit or SimplifiedToolkit()
    tools = toolkit_set.get_tools()
    output_parser = NewAgentOutputFixingParser()

    if not agent_kwargs:
        agent_kwargs = {
            "memory": memory,
            "prefix": PREFIX,
            "format_instructions": FORMAT_INSTRUCTIONS,
            "suffix": SUFFIX,
            "output_parser": output_parser,
            "input_variables": INPUT_VARIABLES,
            "return_intermediate_steps": True,
        }
    # FIXME: preserve the public_html_references from each tool and do not get from LLM
    executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        callbacks=callbacks,
        verbose=verbose,
        agent_kwargs=agent_kwargs,
        **(agent_executor_kwargs or {}),
    )

    return executor
