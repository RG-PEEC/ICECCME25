import json
import math
from typing import List, Dict

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langchain_core.messages.utils import convert_to_messages

import pandas as pd

from assistants.AssistantABC import AssistantABC
from assistants.utils import log_message, extract_esco_obj

from assistants.assistant_models import EscoOverallResult, GPTOverallState
from datasets import get_tokens_of_messages

import os

TOKEN_WINDOW = 5000

api_key = os.environ.get("OPENAI_API_KEY")


class BaselineAssistantCAG(AssistantABC):
    """
    Baseline Assistant class for the LLM.
    """

    def __init__(self, system_prompt_template: str,
                 model_name: str = "qwen2.5-7b-instruct-1m",
                 temperature: float = 0.1,
                 streaming: bool = False,
                 few_shot_examples: List[Dict[str, str]] = None):
        """
        Initialize the Baseline Assistant with the given parameters.
        """
        self.model_name = model_name
        if streaming:
            self.stream_handler = StreamingStdOutCallbackHandler()

        self.prompt_template = system_prompt_template

        if "gpt" in model_name.lower():
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0,
                streaming=streaming,
                callbacks=[self.stream_handler] if streaming else [],
                max_tokens=1000,
                max_retries=2,
                use_responses_api=True
            )
        else:
            self.llm = ChatOpenAI(
                openai_api_base="http://localhost:1234/v1",
                openai_api_key="lm-studio",
                model_name=model_name,
                temperature=temperature,
                streaming=streaming,
                callbacks=[self.stream_handler] if streaming else [],
                max_tokens=4000,
                max_retries=2,
            )
        self.graph_builder = None
        self.graph = None
        self.few_shot_examples = few_shot_examples if few_shot_examples is not None else []
        self.esco_skills = pd.read_csv(
            "/Users/p42939_christophgassner/Code/iceccme_experiments/data/transversalSkillsCollection_en.csv")
        self.esco_skills = self.esco_skills['preferredLabel'].tolist()
        self.build_graph()

    def set_prompt(self, prompt: str) -> None:
        self.prompt_template = prompt

    def invoke(self, state_dict: dict) -> str:
        """
        Invoke the assistant with the given state dictionary.
        """
        state = GPTOverallState(description=state_dict.get("description", ""), results=[])
        result = self.graph.invoke(state)
        return result

    def generate_skill_list(self, state: GPTOverallState) -> GPTOverallState:
        """
        Generate a list of ESCO skills by splitting the master list into N parts,
        querying each segment separately, and aggregating results.
        """
        parser = JsonOutputParser(pydantic_object=EscoOverallResult)
        assistant_role = "assistant"
        # Prepare few-shot messages
        few_shot_messages = []
        for example in self.few_shot_examples:
            few_shot_messages.append({"role": "user",
                                      "content": f"Which ESCO soft-skills (Transversal skills) are in this description?\n {example['description']}",
                                      "example": True})
            few_shot_messages.append({"role": assistant_role,
                                      "content": "{" + f"results: [{self.get_few_shot_str_new(example['output'])}" + "]}",
                                      "example": True})

        # Split ESCO skills into 10 roughly equal chunks
        skills_t = list(self.esco_skills)
        total = len(skills_t)

        total_tokens = get_tokens_of_messages(str(skills_t))
        split_num = max(1, math.ceil((total_tokens / TOKEN_WINDOW)))
        chunk_size = max(1, math.ceil(total / split_num))
        chunks = [skills_t[i:i + chunk_size] for i in range(0, total, chunk_size)]

        system_role = "user" if "mistral" in self.llm.model_name else "system"
        aggregated = []
        # Loop over each chunk and query LLM
        for idx, chunk in enumerate(chunks, start=1):
            messages = [
                # system prompt
                {"role": system_role,
                 "content": self.prompt_template.format(
                     response_format=parser.get_format_instructions()) + "\n" + "Only use labels from the provided list with allowed labels or return an empty list. Just return the JSON-List with the skills from the context, matching the user's provided description." + "Example Ouptut" + """{
  "results": [
    {
      "preferredLabel": "manage digital identity",
      "sentence_piece": "social media management, engagement, potential automation; growing brand awareness, posting",
      "reason": "The task involves managing social media platforms and automating processes, which relates to managing digital identity."
    },
    {
      "preferredLabel": "use equipment, tools or technology with precision",
      "sentence_piece": "social media management, engagement, potential automation; growing brand awareness, posting",
      "reason": "The task requires using social media platforms and possibly automation tools precisely."
    },
    {
      "preferredLabel": "build networks",
      "sentence_piece": "social media management, engagement, potential automation; growing brand awareness, posting",
      "reason": "The task involves engaging with audiences on multiple social media platforms, which relates to building networks."
    },
    {
      "preferredLabel": "work in teams",
      "sentence_piece": "Confronting Domestic Violence is seeking a high energy social media brand awareness volunteer.",
      "reason": "Volunteering for a social media role within an organization suggests working in teams or collaborative environments."
    },
    {
      "preferredLabel": "demonstrate willingness to learn",
      "sentence_piece": "if you as the expert deem it best to have more, we will do it!",
      "reason": "The willingness to adapt and learn new skills or tools is implied in the openness to additional responsibilities."
    }
  ]
}"""},
                {"role": system_role,
                 "content": f"Only return labels defined in the following list, if no skills are matching the provided "
                            f"description, return an empty list, not containing skills at all. Only take labels once "
                            f"(do not repeat them) from this list of allowed labels: ({idx}/{len(chunks)}): {chunk}"},
                # examples
                *few_shot_messages,
                # request
                {"role": "user",
                 "content": f"Which ESCO soft-skills (Transversal skills) are in this description?\n: {state['description']}"}
            ]
            messages = convert_to_messages(messages)
            log_message(messages, ("BaselineAssistantCAGFewShot" if len(
                self.few_shot_examples) > 0 else "BaselineAssistantCAG") + "_" + self.llm.model_name)

            response = self.llm.invoke(messages)
            log_message([response], ("BaselineAssistantCAGFewShot" if len(
                self.few_shot_examples) > 0 else "BaselineAssistantCAG") + "_" + self.llm.model_name)
            if type(response.content) == list:
                response.content = response.content[0]

            if type(response.content) == dict:
                response.content = response.content.get("text", response.content)

            # Parse response
            try:
                skills = parser.parse(response.content)
                result = skills.get("results", response.content)
            except Exception as e:
                try:
                    skills = extract_esco_obj(response.content, key="results")
                    result = skills.get("results")
                except Exception as e:
                    try:
                        json_result = json.loads(response.content)
                        result = json_result.get("results", response.content)
                    except Exception as e:
                        try:
                            json_response = response.content
                            text = json_response.get("text")
                            text = json.loads(text[0])
                            result = text.get("results", response.content)
                        except Exception as e:
                            result = [{"Error": 1, "response": response.content}]
            # Merge

            aggregated += result if result is not None else []

        return GPTOverallState(
            description=state["description"],
            results=aggregated
        )

    def build_graph(self):
        """
        Build the state graph.
        """
        self.graph_builder = StateGraph(GPTOverallState)
        self.graph_builder.add_node("generate_skills", self.generate_skill_list)
        self.graph_builder.add_edge(START, "generate_skills")
        self.graph_builder.add_edge("generate_skills", END)
        self.graph = self.graph_builder.compile()
