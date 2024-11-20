"""Callback Handler that prints to std out."""
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.input import print_text
from langchain.schema import AgentAction, AgentFinish, LLMResult

# Initialize the logger module
import logging
logger = logging.getLogger(__name__)


class StdOutAllCallbackHandler(BaseCallbackHandler):
    """Callback Handler that prints to std out."""

    def __init__(self, color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        self.color = color

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        print("\033[95m## [on_llm_start]\033[0m")
        print("   \033[1mPrompts:\033[0m")
        for prompt in prompts:
            print("      \033[1m" + prompt.replace("\n", " ") + " \033[0m")

        print("   \033[1mSerialized:\033[0m")
        for key, value in serialized.items():
            print(f"\033[1m      {key}: {value} \033[0m")
        logger.info("[on_llm_start]: " + str(prompts))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("\033[95m## [on_llm_end]\033[0m")
        print("   \033[1mResult:\033[0m")
        print(f"      \033[1m{response}\033[0m")
        logger.info("[on_llm_end]: " + str(response))

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print("\033[95m## [on_llm_new_token]\033[0m")
        print("   \033[1mToken:\033[0m")
        print(f"      \033[1m{token}\033[0m")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        print("\033[95m## [on_llm_error]\033[0m")
        print("   \033[1mError:\033[0m")
        print(f"      \033[1m{error}\033[0m")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""

        print("\033[95m## [on_chain_start]\033[0m")
        print("   \033[1mInputs:\033[0m")
        print(f"      \033[1m{inputs}\033[0m")
        print("   \033[1mSerialized:\033[0m")
        print(f"      \033[1m{serialized}\033[0m")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        print("\033[95m## [on_chain_end]\033[0m")
        print("   \033[1mOutput:\033[0m")
        print(f"      \033[1m{outputs}\033[0m")

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        print("\033[95m## [on_chain_error]\033[0m")
        print("   \033[1mError:\033[0m")
        print(f"      \033[1m{error}\033[0m")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        print("\033[95m## [on_tool_start]\033[0m")
        print("   \033[1mInputs:\033[0m")
        print(f"      \033[1m{input_str}\033[0m")
        print("   \033[1mSerialized:\033[0m")
        print(f"      \033[1m{serialized}\033[0m")

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        print("\033[95m## [on_tool_start]\033[0m")
        print_text("   " + action.log, color=color if color else self.color)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        print("\033[95m## [on_tool_end]\033[0m")
        """If not the final action, print out observation."""
        if observation_prefix is not None:
            print_text(f"\n   {observation_prefix}")
        print_text("   " + output, color=color if color else self.color)
        if llm_prefix is not None:
            print_text(f"\n   {llm_prefix}")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        print("\033[95m## [on_tool_error]\033[0m")
        print("   \033[1mError:\033[0m")
        print(f"      \033[1m{error}\033[0m")

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        print("\033[95m## [on_text]\033[0m")
        """Run when agent ends."""
        print_text("   " + text, color=color if color else self.color, end=end)

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        print("\033[95m## [on_agent_finish]\033[0m")
        """Run on agent end."""
        print_text("   " + finish.log, color=color if self.color else color, end="\n")
