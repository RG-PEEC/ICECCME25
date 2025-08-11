from langchain_core.messages import convert_to_messages
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from assistants import log_message
from assistants.AssistantABC import AssistantABC
from assistants.assistant_models import EscoOverallResult, GPTOverallState

from typing import List, Dict

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

import os

# get api key for openai
api_key = os.environ.get("OPENAI_API_KEY")


class ChatgptAssistant(AssistantABC):
    def __init__(self, system_prompt_template: str,
                 model_name: str = "o4-mini-2025-04-16",#"gpt-4.1-2025-04-14",
                 temperature: float = 1,
                 streaming: bool = False,
                 few_shot_examples: List[Dict[str, str]] = None):
        """
        Initialize the Baseline Assistant with the given parameters.
        """
        # add streamhandler if needed
        if streaming:
            self.stream_handler = StreamingStdOutCallbackHandler()

        # define prompt template for system prompt
        self.prompt_template = system_prompt_template

        # initialize llm
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            streaming=streaming,
            callbacks=[self.stream_handler] if streaming else [],
            max_tokens=8000,
            max_retries=2,
        )

        self.graph_builder = None
        self.graph = None
        self.few_shot_examples = few_shot_examples if few_shot_examples is not None else []
        self.build_graph()


    def set_prompt(self, prompt: str) -> None:
        pass

    def build_graph(self) -> None:
        """
        Build the state graph for the assistant.
        """
        self.graph_builder = StateGraph(GPTOverallState)
        self.graph_builder.add_node("generate_skills", self.generate_skill_list)

        self.graph_builder.add_edge(START, "generate_skills")
        self.graph_builder.add_edge("generate_skills", END)

        self.graph = self.graph_builder.compile()

    def invoke(self, state_dict: dict) -> str:
        state = GPTOverallState(description=state_dict.get("description", ""), results=[])
        result = self.graph.invoke(state)
        return result

    def generate_skill_list(self, state: GPTOverallState) -> GPTOverallState:
        # define parser for Output
        parser = JsonOutputParser(pydantic_object=EscoOverallResult)

        # create few shot messages
        few_shot_messages = []
        for example in self.few_shot_examples:
            few_shot_messages.append({"role": "user", "content": f"description: {example['description']}"})
            few_shot_messages.append({"role": "assistant",
                                      "content": "{" + f"(Context:) ESCO-Skills (Candidates): {example['output']}" + "}"})

        # build message list for LLM call
        messages = [
            # system prompt
            {"role": "system",
             "content": self.prompt_template.format(response_format=parser.get_format_instructions())},
            # examples
            *few_shot_messages,
            # request
            {"role": "user", "content": f"description: {state['description']}"}
        ]
        messages = convert_to_messages(messages)
        log_message(messages, "ChatGPTAssistant")

        response = self.llm.invoke(messages)
        log_message([response], "ChatGPTAssistant")

        # parse response
        try:
            skills = parser.parse(response.content)
            # skills = skills.get("escoSkills", response.content)
        except Exception as e:
            skills = ["Error"]

        return GPTOverallState(
            description=state['description'],
            results=skills
        )