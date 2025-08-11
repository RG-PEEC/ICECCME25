from . import AssistantABC
from .baseline.BaselineAssistantClass import BaselineAssistant
from .baseline.BaselineAssistantClassCAG import BaselineAssistantCAG
from .chatGPT.chatgpt_api_assistant import ChatgptAssistant
from .rag.RagAssistant import RagAssistant

"""["BaselineAssistant", "BaselineAssistantFewShot",
                                "RagAssistant", "RagAssistantFewShot",
                                "ApiAssistant", "ApiAssistantFewShot",
                                # "EmbeddingAssistant", "EmbeddingAssistantFewShot",
                                ]"""


def get_assistant(assistant_str: str, prompt: str, model_name: str, few_shot_examples=None, temperature=0) -> AssistantABC:
    """
    Get the assistant class based on the string name.
    """
    assistant = None
    if "BaselineAssistant" in assistant_str:
        assistant = BaselineAssistant(system_prompt_template=prompt,
                                      model_name=model_name,
                                      few_shot_examples=few_shot_examples,
                                      temperature=temperature)
    elif "RagAssistant" in assistant_str:
        assistant = RagAssistant(system_prompt_template=prompt,
                                 model_name=model_name,
                                 few_shot_examples=few_shot_examples,
                                 temperature=temperature)
    elif "GPTAssistant" in assistant_str:
        assistant = ChatgptAssistant(system_prompt_template=prompt, few_shot_examples=few_shot_examples)
    elif "EmbeddingAssistant" in assistant_str:
        pass
    elif "CagAssistant" in assistant_str:
        assistant = BaselineAssistantCAG(system_prompt_template=prompt,
                                         model_name=model_name,
                                        few_shot_examples=few_shot_examples,
                                        temperature=temperature)

    return assistant