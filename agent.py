from typing import Any
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent

from utility.utils import get_openai_api_key

prompt = "you are an ai assistant and your name is jeff"


class EscroAgent():

    def __init__(self, llm=None, tools=None):

        if not llm:
            self.llm = OpenAI(model="gpt-3.5-turbo",
                              api_key=get_openai_api_key(),
                              temperature=0.1)
        else:
            self.llm = llm

        self.tools = tools

        self.agent = ReActAgent.from_tools(
            self.tools,
            llm=self.llm,
            verbose=True,
            s=prompt
        )

    def __call__(self, query) -> Any:
        return self.agent.chat(query)


if __name__ == "__main__":

    agent = EscroAgent()
    print(agent)
    print(agent("hello, what's your name?"))
