import ast
from abc import ABC, abstractmethod

class AssistantABC(ABC):
    """
    Abstract base class for assistants.
    """

    @abstractmethod
    def set_prompt(self, prompt: str) -> None:
        pass

    @abstractmethod
    def build_graph(self) -> None:
        pass

    @abstractmethod
    def invoke(self, state_dict: dict) -> str:
        pass

    def get_few_shot_str_new(self, example_output: str) -> str:
        """
        Get the few shot string for the new example.
        """
        example_list = ast.literal_eval(example_output)
        example_list = ["{'preferredLabel: "+ str(item) + ",\n sentence_piece: '<sentence_piece>',\n reason: '<reason>'}" for item in example_list if isinstance(item, str)]
        return ", ".join(example_list)