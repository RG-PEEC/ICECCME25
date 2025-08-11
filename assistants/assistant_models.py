from typing import List, TypedDict, Dict, List

from langchain_core.documents import Document
from pydantic import BaseModel, Field
class LLMResponse(BaseModel):
    escoSkills: List[str] = Field(description="List of ESCO skills")


class InputState(TypedDict):
    description: str


class OutputState(TypedDict):
    escoSkills: List[str]


class MessageOverallState(InputState, OutputState):
    pass


class RagRetrivalState(TypedDict):
    description: str
    retrival_queries: List[str]
    retrival_response: List[List[Document]]


class RagQueryState(TypedDict):
    description: str
    retrival_queries: List[str]


class RagQueryResponseState(BaseModel):
    retrival_queries: List[str] = Field(description="List of Search Queries to find ESCO Skills")


class EscoOverallResult(BaseModel):
    class EscoSkillResult(BaseModel):
        preferredLabel: str = Field(description="The preferred label of the ESCO skill")
        sentence_piece: str = Field(description="The sentence piece which contains the ESCO skill")
        reason: str = Field(description="The reason why the ESCO skill occurs in the sentence piece")
    results: List[EscoSkillResult] = Field(description="List of ESCO skills with their preferred labels and reasons")

class GPTOutputState(TypedDict):
    results: List[EscoOverallResult.EscoSkillResult]

class GPTOverallState(InputState, GPTOutputState):
    pass