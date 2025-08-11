# general imports
import json
import math
from typing import List, Dict, TypedDict, Set

# RAG specific imports
import chromadb
from langchain_chroma import Chroma
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import LLMChainExtractor

# LangChain specific imports
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Embeddings is RAG specific
from langchain.schema import SystemMessage, messages_from_dict, HumanMessage
from langchain_core.messages.utils import convert_to_messages

# LangGraph specific imports
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

# Implementation specific imports
from assistants.utils import log_message, extract_esco_obj
from assistants.assistant_models import (GPTOverallState,
                                         EscoOverallResult,
                                         RagRetrivalState,
                                         RagQueryState,
                                         RagQueryResponseState)
from assistants.AssistantABC import AssistantABC
from datasets import get_tokens_of_messages

import os

TOKEN_WINDOW = 4000
api_key = os.environ.get("OPENAI_API_KEY")


class RagAssistant(AssistantABC):
    def __init__(self, system_prompt_template: str,
                 model_name: str = "qwen2.5-7b-instruct-1m",
                 temperature: float = 0.0,
                 streaming: bool = False,
                 few_shot_examples: List[Dict[str, str]] = None):
        self.model_name = model_name
        if streaming:
            self.stream_handler = StreamingStdOutCallbackHandler()
        self.llm = None
        if "gpt" in model_name.lower():
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                streaming=streaming,
                callbacks=[self.stream_handler] if streaming else [],
                max_tokens=8000,
                max_retries=2,
                use_responses_api=True,
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

        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-all-minilm-l6-v2-embedding",
                                                 openai_api_base="http://localhost:1234/v1",
                                                 api_key='lm-studio',
                                                 check_embedding_ctx_length=False)
        self.persistent_chroma_client = chromadb.PersistentClient(
            path="/Users/p42939_christophgassner/Code/iceccme_experiments/assistants/rag/data/emb")

        self.vectorstore: Chroma = Chroma(
            client=self.persistent_chroma_client,
            collection_name="esco_embeddings",
            embedding_function=self.embeddings_model,
            persist_directory="/Users/p42939_christophgassner/Code/iceccme_experiments/assistants/rag/data/emb"
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        self.prompt_template = system_prompt_template
        self.compressor = LLMChainExtractor.from_llm(self.llm)
        self.compression_retriever = DocumentCompressorPipeline(transformers=[self.compressor])

        self.graph_builder = None
        self.graph: CompiledStateGraph = None
        self.few_shot_examples = few_shot_examples if few_shot_examples is not None else []
        self.build_graph()

    def set_prompt(self, prompt: str) -> None:
        """
        Set the prompt for the assistant.
        """
        self.prompt_template = prompt
        self.build_graph()

    def invoke(self, state_dict: dict) -> str:
        """
        Invoke the assistant with the given state dictionary.
        """
        result = self.graph.invoke(GPTOverallState(**state_dict))
        return result

    def generate_retriever_queries_(self, state: GPTOverallState) -> RagQueryState:
        desc = state["description"]
        # split every 5 sentences
        sentences = desc.split(". ")
        split_sentences = []
        for i in range(0, len(sentences), 10):
            split_sentences.append(". ".join(sentences[i:i + 10]))

        return RagQueryState(description=state["description"], retrival_queries=split_sentences)

    def generate_retriever_queries(self, state: GPTOverallState) -> RagQueryState:
        """Erzeuge die Abfrage für den Retriever."""
        query_list_parser = JsonOutputParser(pydantic_object=RagQueryResponseState)

        search_query_prompt = (f"We are trying to find all explicit and implied Skills and Knowledge to fulfill "
                               f"the give job or volunteering opportunities. For this we have to call the ESCO-Database "
                               f"and need search-queries. We are looking for all (really all) mentioned hard-skills "
                               f"and soft-skills and transversal-skills and knowledge in the given Description. "
                               f"Please generate a list with all possible search-queries you can think of, to retrieve "
                               f"from the ESCO-Datebase. Esco ist the EU-Skill Framework named: European Occupations, "
                               f"Skills, Competences and Qualifications. Analyse every sentence in detail to really "
                               f"find all skills. Your Task is to generate a list of search queries to find all skills in a taks description. Here is the Task-Description: " +
                               f"{state["description"]}" + "/n Please respond with a JSON Object described in the following " +
                               f"output format: {query_list_parser.get_format_instructions()} \n Maximum 40 different search queries!!!")
        # extrahiere nur die Skill‑Bezeichnungen
        log_message([HumanMessage(content=search_query_prompt)], ("RagAssistantFewShot" if len(self.few_shot_examples) > 0 else "RagAssistant")+"_"+self.llm.model_name)
        response = self.llm.invoke(search_query_prompt)
        # response_message = messages_from_dict([response])[0]
        log_message([response], ("RagAssistantFewShot" if len(self.few_shot_examples) > 0 else "RagAssistant")+"_"+self.llm.model_name)
        desc = state["description"]
        # split every 5 sentences
        sentences = desc.split(". ")
        split_sentences = []
        for i in range(0, len(sentences), 10):
            split_sentences.append(". ".join(sentences[i:i + 10]))
        try:
            query_list = query_list_parser.parse(response.content)
            query_list["retrival_queries"] = query_list["retrival_queries"] + split_sentences
        except Exception as e:
            query_list = {"retrival_queries": split_sentences}

        return RagQueryState(description=state["description"], retrival_queries=query_list["retrival_queries"])


    def rag_retrieve(self, state: RagQueryState) -> RagRetrivalState:
        """Hole relevante Skill‑Dokumente via Retriever."""
        log_message([SystemMessage(content="RAG Retrieve for: " + str(state["retrival_queries"]))], ("RagAssistantFewShot" if len(self.few_shot_examples) > 0 else "RagAssistant")+"_"+self.llm.model_name)
        retrival_response = [self.retriever.invoke("" + desc, k=5) for desc in state["retrival_queries"]]
        retrival_response.append(self.retriever.invoke("" + state["description"], k=10))
        # flatten retrival response
        retrival_response = [item for sublist in retrival_response for item in sublist]
        retrival_response = [{doc.page_content.replace("ESCO preferredLabel: ", "") : doc.metadata.get("description", "")} for doc in retrival_response]
        # retrival_response_str = "".join([str(d.to_json()) for d in retrival_response])
        # make retrival_response unique
        unique = list({str(doc) for doc in retrival_response})
        log_message([SystemMessage(content=retrival_response)], ("RagAssistantFewShot" if len(self.few_shot_examples) > 0 else "RagAssistant")+"_"+self.llm.model_name)
        return RagRetrivalState(description=state["description"],
                                retrival_response=unique,
                                retrival_queries=state["retrival_queries"])


    def generate_response(self, state: RagRetrivalState) -> GPTOverallState:
        """LLM fasst die Treffer zusammen, filtert Doppler, gibt JSON aus."""
        parser = JsonOutputParser(pydantic_object=EscoOverallResult)

        few_shot_messages = []
        assistant_role = "assistant"
        for example in self.few_shot_examples:
            few_shot_messages.append({"role": "user",
                                      "content": f"Which hard- and soft-skills and knowledge are in this description?\n {example['description']} \n Please only use Skills provided from the following (Esco-Skills-)Context: [...]",
                                      "example": True})
            few_shot_messages.append(
                {"role": assistant_role, "content": "{" + f"result: [{self.get_few_shot_str_new(example['output'])}]" + "}", "example": True})

        result_splits = []
        total_tokens = get_tokens_of_messages(str(state['retrival_response']))
        split_num = max(1, math.ceil((total_tokens / TOKEN_WINDOW)))
        chunk_size = max(1, math.ceil(len(state['retrival_response']) / split_num))

        for i in range(0, len(state['retrival_response']), chunk_size):
            result_splits.append(state['retrival_response'][i:i + chunk_size])

        all_skills = []
        system_role = "user" if "mistral" in self.llm.model_name else "system"
        for split in result_splits:
            # build message list for LLM call
            messages = [
                # system prompt
                {"role": system_role,
                 "content": self.prompt_template.format(response_format=parser.get_format_instructions()) + "\n DO NOT REPEAT ANY SKILL!!!!" + "Example Ouptut" + """{
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
                # examples
                *few_shot_messages,
                # request
                {"role": "user",
                 "content": f"Please only use the preferredLabel of Esco-Skills in this Context/Skill Candidates: {split}"},
                {"role": "user",
                 "content": f"Which ESCO soft-skills (Transversal skills) are in this description?\n: {state['description']}"+ "\n Please do not repeat any skills. Every Skills is only allowed once per Task-Description."},
            ]
            messages = convert_to_messages(messages)
            log_message(messages, ("RagAssistantFewShot" if len(self.few_shot_examples) > 0 else "RagAssistant")+"_"+self.llm.model_name)
            response = self.llm.invoke(messages)
            # response_message = messages_from_dict([response])[0]
            log_message([response], ("RagAssistantFewShot" if len(self.few_shot_examples) > 0 else "RagAssistant")+"_"+self.llm.model_name)

            # Parse response
            if type(response.content) == list:
                response.content = response.content[0]

            if type(response.content) == dict:
                response.content = response.content.get("text", response.content)

            try:
                skills = parser.parse(response.content)
                result = skills.get("results", response.content)
            except Exception as e:
                try:
                    skills = extract_esco_obj(response.content, key="results")
                    result = skills.get("results")
                except Exception as e:
                    try:
                        json_response = response.content[0]
                        text = json_response.get("text")
                        text = json.loads(text)
                        result = text.get("results", response.content)
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
            all_skills += result if result is not None else []
        return GPTOverallState(description=state["description"], results=all_skills)


    def build_graph(self) -> None:
        """
        Build the graph for the assistant.
        """
        self.graph_builder = StateGraph(GPTOverallState)
        self.graph_builder.add_node("generate_retriever_queries", self.generate_retriever_queries)
        self.graph_builder.add_node("rag_retrieve", self.rag_retrieve)
        self.graph_builder.add_node("generate_response", self.generate_response)

        self.graph_builder.add_edge(START, "generate_retriever_queries")
        self.graph_builder.add_edge("generate_retriever_queries", "rag_retrieve")
        self.graph_builder.add_edge("rag_retrieve", "generate_response")
        self.graph_builder.add_edge("generate_response", END)

        self.graph = self.graph_builder.compile()
