from transformers import AutoTokenizer

"""
from langchain_core.output_parsers import JsonOutputParser
from tqdm import tqdm

from assistants import LLMResponse
from job.JobDataset import jobDataset
from volunteer.VolunteerDataset import volunteerDataset
from prompting import PromptModel

experiment_datasets = [jobDataset, volunteerDataset]

promptModel = PromptModel()
prompts = promptModel.prompts
parser = JsonOutputParser(pydantic_object=LLMResponse)

token_count_dict = {}

for dataset in tqdm(experiment_datasets, desc="Datasets", position=0):
    # load data
    X, y = dataset.get_data()
    for prompt in tqdm(prompts, desc="Prompts", position=3, leave=False):
        input_data, eval_data = dataset.get_data()
        for (elem) in input_data:
            few_shot_examples = dataset.get_few_shot_examples()
            few_shot_messages = []
            for example in few_shot_examples:
                few_shot_messages.append({"role": "user", "content": f"description: {example['description']}"})
                few_shot_messages.append(
                    {"role": "assistant", "content": "{" + f"escoSkills: {example['output']}" + "}"})

            # build message list for LLM call
            messages = [
                # system prompt
                {"role": "system",
                 "content": prompt.get_text().format(response_format=parser.get_format_instructions())},
                # examples
                *few_shot_messages,
                # request
                {"role": "user", "content": f"description: {elem}"},
            ]

            flat_prompt = ""
            for msg in messages:
                flat_prompt += f"{msg['role']}: {msg['content']}\n"

            # print(flat_prompt)
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct",
                                                      trust_remote_code=True)
            tokens = tokenizer.tokenize(flat_prompt)

            token_count = len(tokens)
            key = f"{dataset.__class__.__name__}_{prompt.name}_{elem[:18]}"

            token_count_dict[key] = token_count
            print(f"Prompt: {key}, Token Count: {token_count}")

# print longest prompt
longest_prompt = max(token_count_dict, key=token_count_dict.get)
print(f"Longest prompt: {longest_prompt} with {token_count_dict[longest_prompt]} tokens")"""

def get_tokens_of_messages(messages) -> int:
    """
    Get the number of tokens in a list of messages.
    :param messages: List of messages
    :return: Number of tokens
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct", trust_remote_code=True)
    if isinstance(messages, str):
        # If the input is a string, tokenize it directly
        tokens = tokenizer.tokenize(messages)
        return len(tokens)

    flat_prompt = ""
    for msg in messages:
        flat_prompt += f"{msg['role']}: {msg['content']}\n"
    tokens = tokenizer.tokenize(flat_prompt)
    return len(tokens)