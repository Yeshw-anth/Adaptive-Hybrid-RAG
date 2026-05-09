from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any

class QueryMetadata(BaseModel):
    normalized_query: str
    keyword_tokens: List[str] = Field(default_factory=list)
    query_type: Literal["simple", "complex", "analytical", "comparative", "keyword"] = Field(default="simple")
    intent: Literal["fact-seeking", "summary", "comparison", "causal-analysis"] = Field(default="fact-seeking")
    complexity: Literal["low", "medium", "high"] = Field(default="low")
    keywords: List[str] = Field(default_factory=list)
    suggested_depth: int = Field(default=4)
    expected_answer_format: Literal["list", "single_value", "explanation", "code_snippet", "table"] = Field(default="explanation")
    content_hints: List[Literal["table", "code", "text", "graph"]] = Field(default_factory=list)
    retrieval_strategy: Literal["vector", "graph", "hybrid", "hybrid_graph"] = Field(default="vector")
    max_latency: float | None = Field(default=None, description="Maximum allowed latency in seconds")
    max_cost: float | None = Field(default=None, description="Maximum allowed cost in USD")

class Strategy(BaseModel):
    pipeline: Literal["fast", "accurate", "code", "structured", "keyword"]
    retrieval_strategy: Literal["vector", "hybrid"]
    use_reranker: bool
    use_parent_child: bool = False
    model: str
    top_k: int

class DocumentMetadata(BaseModel):
    file_name: str | None = None
    chunk_id: str | None = None
    chunk_sequence_number: int | None = None
    word_count: int | None = None
    content_hash: str | None = None
    file_type: str | None = None
    file_size: int | None = None
    section_title: str | None = None
    is_parent: bool = False
    parent_id: str | None = None
    content_type: str | None = None
    structured_data_json: str | None = None
    has_structure: bool = False
    image_path: str | None = None
    # You can add other fields from the original document's metadata here
    # For example: source, author, creation_date, etc.

class Document(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float | None = Field(default=None, description="The relevance score of the document.")

class OutputLog(BaseModel):
    query_id: str
    query: str
    query_metadata: QueryMetadata
    selected_strategy: Strategy
    final_answer: str
    final_context: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document] | None = None
    final_docs: List[Document]
    latency: float
    confidence_score: float | None = None
    action_taken: str | None = None
    expansion_triggered: bool = False
    timestamp: str
    # RAGas metrics
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None