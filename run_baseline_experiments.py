import time

from experiment import Experiment
from tqdm import tqdm
from utils import MongoHandler
from datetime import datetime
from assistants import get_assistant
from prompting import PromptModel, Prompt
from datasets import volunteerDataset, jobDataset, JobDataset, VolunteerDataset, volunteerDatasetSoft, \
    VolunteerDatasetSoft
from ast import literal_eval
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKEN_WINDOW = "7000"

import subprocess
import logging

logger_chat_baseline = logging.getLogger("chat_baseline")
logger_chat_baseline.setLevel(logging.INFO)
mongo_handler_chat = MongoHandler("ChatBaseline", "ChatBaseline")
formatter_chat = logging.Formatter('%(asctime)s %(message)s')
logger = logging.getLogger("RunExperiment")
logger.setLevel(logging.INFO)
mongo_handler = MongoHandler("Experiment", "ExperimentLog")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
mongo_handler.setFormatter(formatter)
logger.addHandler(mongo_handler)

model_paths = [ # "gpt-4.1-2025-04-14",
    # "gpt-4.1-nano-2025-04-14", #"gpt-4o-mini-2024-07-18", "o4-mini-2025-04-16", "gpt-4.1-nano-2025-04-14", #,
    "lmstudio-community/Qwen2.5-14B-Instruct-GGUF/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
    # "lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf",
    # "lmstudio-community/Mistral-Small-3.1-24B-Instruct-2503-GGUF/Mistral-Small-3.1-24B-Instruct-2503-Q3_K_L.gguf",
    # "lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    # "lmstudio-community/phi-4-GGUF/phi-4-Q4_K_M.gguf",
    # "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    # "lmstudio-community/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf",
    # "lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
]


def get_model_name_from_path(model_path: str) -> str:
    """
    Get the model name from the model path.
    :param model_path: The model path.
    :return: The model name.
    """
    if "QwQ-32B" in model_path:
        return "qwq-32b"
    elif "gemma" in model_path:
        return "gemma-2-2b-it"
    elif "DeepSeek" in model_path:
        return "deepseek-r1-distill-qwen-7b"
    elif "phi" in model_path:
        return "phi-4"
    elif "Qwen3" in model_path:
        return "qwen3-14b"
    elif "Mistral-7B" in model_path:
        return "mistral-7b-instruct-v0.3"
    elif "Mistral-Small" in model_path:
        return "mistral-small-3.1-24b-instruct-2503"
    elif "llama3" in model_path:
        return "meta-llama-3.1-8b-instruct"
    elif "gpt-4.1-nano-2025-04-14" in model_path:
        return "gpt-4.1-nano-2025-04-14"
    elif "gpt-4o-mini-2024-07-18" in model_path:
        return "gpt-4o-mini-2024-07-18"
    elif "o4-mini-2025-04-16" in model_path:
        return "o4-mini-2025-04-16"
    elif "gpt-4.1-2025-04-14" in model_path:
        return "gpt-4.1-2025-04-14"
    else:
        return "qwen2.5-14b-instruct" if "Instruct-1M" not in model_path else "qwen2.5-14b-instruct-1m"


def is_model_loaded(model_path: str) -> bool:
    result = subprocess.run(
        ["lms", "ps"],
        capture_output=True,
        text=True
    )
    return model_path in result.stdout


experiment_datasets = [volunteerDataset]  # volunteerDatasetSoft]#jobDataset]  # ,
experiment_assistants: [str] = [# "ApiAssistant", "ApiAssistantFewShot",
                                "CagAssistant",
                                "CagAssistantFewShot",
                                "RagAssistantFewShot", "RagAssistant",
                                "BaselineAssistantFewShot", "BaselineAssistant",
                                # "GPTAssistantFewShot", "GPTAssistant",
                                # "EmbeddingAssistant", "EmbeddingAssistantFewShot",
                                ]

promptModel = PromptModel()

"""for prompt in promptModel.prompts:
    print(prompt.get_text())"""

prompts = promptModel.get_prompt_by_name("Prompt_may_11")

PROMPT_GET_TRANSVERSAL_ESCO_SKILLS = "You are a friendly assistant which is able to extract EXPLICIT mentioned Transversal ESCO-skills (out of a given set) from a given task/job description. Transversal Esco Skills are skills within the ESCO Ontology, you will find all transversal esco skills preferredLabel's with a descprion of them as attachement to this Message. Your task is to extract ALL of these EXPLICIT MENTIONED and also IMPLICIT Mentioned transversal esco skills (the preferredLabel) from a given job description and return them in a JSON Object with the schema defined at the end of this message. Please do only map especially mentioned transversal skills. Please try getting as much as possible (Up to 20 or even more)"
# prompts = #[prompts]
prompts.name = "TransversalSkillsReasoning"
prompts.input_description = ""
# "\nThis are all Transversal ESCO Skills, you will see the preferredLabel's : with the official meaning "
# "of this label: ["+transversal_str+"]\n\n" + PROMPT_GET_TRANSVERSAL_ESCO_SKILLS +
prompts.task_instructions = (PROMPT_GET_TRANSVERSAL_ESCO_SKILLS + "Note that in every Sentence of the Posting, could be "
                                                                  "more than one transversal skill. You should identify "
                                                                  "them separately in the following " + "Output Format: "
                                                                  "'{response_format}' Please provide the output in the "
                                                                  "specified format, but not as schema itself..\n\n")

transversal_df = pd.read_csv("/Users/p42939_christophgassner/Code/CONDA_ICECCME25/data/transversalSkillsCollection_en.csv")

transversal_list = transversal_df['preferredLabel'].tolist()
transversal_str = ";".join(transversal_list)

prompts = [prompts]


def get_dataset_class_name(dataset):
    if isinstance(dataset, JobDataset):
        return "JobDataset"
    elif isinstance(dataset, VolunteerDataset):
        return "VolunteerDataset"
    elif isinstance(dataset, VolunteerDatasetSoft):
        return "VolunteerDatasetSoft"
    else:
        raise ValueError("Unknown dataset class")


def load_model(model_path: str, is_few_shot: bool = False) -> bool:
    """
    Load the model from the given path.
    :param model_path: The model path.
    :return: True if the model is loaded, False otherwise.
    """
    token_win = TOKEN_WINDOW if not is_few_shot else (int(TOKEN_WINDOW) + 2000)
    result = subprocess.run(
        ["lms", "load", model_path, "--context-length", token_win, "--ttl", "300", "--gpu", "max", "--exact"],
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def unload_model(model_path: str) -> bool:
    """
    Unload the model from the given path.
    :param model_path: The model path.
    :return: True if the model is unloaded, False otherwise.
    """
    result = subprocess.run(
        ["lms", "unload", model_path],
        capture_output=True,
        text=True
    )
    return result.returncode == 0


for dataset in tqdm(experiment_datasets, desc="Datasets", position=0):
    # load data
    X, y = dataset.get_data()
    for model in tqdm(model_paths, desc="Models", position=1, leave=False):
        # load model
        if "gpt" in model.lower():
            tqdm.write(f"No loading needed for ChatGPT")
        elif is_model_loaded(model):
            tqdm.write(f"Model {model} already loaded")
        else:
            tqdm.write(f"Loading model {model}")
            load_model(model)
            """            subprocess.run(f"lms load {model} --context-length 5000 --tt"
                           f"l 300 --gpu max --exact",
                           shell=False,
                           executable='/bin/zsh',
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           )"""
        # run prompts
        for assistant_str in tqdm(experiment_assistants, desc="Assistants", position=2, leave=False):
            for prompt in tqdm(prompts, desc="Prompts", position=3, leave=False):
                # print(f"Prompt Title: {prompt.name}")
                tqdm.write(f"Prompt Title: {prompt.name}")
                if "FewShot" in assistant_str:
                    assistant_obj = get_assistant(prompt=prompt.get_text(),
                                                  model_name=get_model_name_from_path(model),
                                                  few_shot_examples=dataset.get_few_shot_examples(),
                                                  assistant_str=assistant_str
                                                  # temperature=0.0
                                                  )
                else:
                    assistant_obj = get_assistant(prompt=prompt.get_text(),
                                                  model_name=get_model_name_from_path(model),
                                                  assistant_str=assistant_str,
                                                  # temperature=0.0
                                                  )
                # filename = f"T0_{assistant_str}_{prompt.name}_{get_dataset_class_name(dataset)}_{get_model_name_from_path(model)}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                filename = f"{assistant_str}_{get_model_name_from_path(model)}_{datetime.now().strftime('%Y_%m_%d-%H:%M:%S')}"
                tqdm.write(f"Running experiment: {filename}")
                experiment = Experiment(assistant_obj, X, y, filename)
                experiment.run_experiment()
                unload_model(model)
                time.sleep(5)