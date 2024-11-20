from langchain.schema import AgentAction as AgentAction_
from langchain.schema import AgentFinish as AgentFinish_


class AgentAction(AgentAction_):
    def __str__(self) -> str:
        return f"Used tool {self.tool} with input {self.tool_input}."


class AgentFinish(AgentFinish_):
    def __str__(self) -> str:
        return f"Finished with return values {self.return_values}."
