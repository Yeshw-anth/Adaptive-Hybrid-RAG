import time
from src.core.logging_config import logger
import pandas as pd
import json
from typing import Any, Dict, List

from src.core.pipelines.base import Pipeline
from src.core.llm.ollama_client import OllamaClient
from src.core.retrieval.retriever import Retriever
from src.core.decision.confidence_engine import ConfidenceEngine


class StructuredPipeline(Pipeline):
    """
    A RAG pipeline that leverages structured data (from tables in documents)
    by generating and executing pandas code to answer queries.
    """

    def __init__(self, llm_client: OllamaClient, retriever: Retriever, confidence_engine: ConfidenceEngine):
        self.llm_client = llm_client
        self.retriever = retriever
        self.confidence_engine = confidence_engine

    def _generate_pandas_code_prompt(self, query: str, df_schema: str, df_head: str) -> str:
        """Generates a prompt for the LLM to write pandas code."""
        return f"""You are a data analysis expert. Your task is to answer a user's query by generating a single line of Python code that uses a pandas DataFrame named 'df'.

The DataFrame has the following schema:
{df_schema}

Here are the first 5 rows of the data:
{df_head}

User Query: "{query}"

Based on the query and the data schema, generate a single, executable line of pandas code to get the answer.
- The code must operate on a DataFrame named 'df'.
- The output of your code should be a pandas DataFrame or a Series.
- DO NOT include any explanations, comments, or any text other than the code itself.
- For example, if the user asks "what is the average revenue?", your output should be something like: df[df['revenue'] > 1000] or df['revenue'].mean()

Pandas Code:
"""

    def _execute_pandas_code(self, code: str, df: pd.DataFrame) -> Any:
        """Safely executes the generated pandas code."""
        try:
            # A more secure environment would use a library like 'asteval'
            # For this project, we'll use a restricted eval
            result = eval(code, {"pd": pd, "df": df})
            return result
        except Exception as e:
            logger.error(f"Error executing generated pandas code: {e}")
            raise

    async def execute(self, query: str, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"Running StructuredPipeline for query: '{query}'")

        # 1. Retrieve relevant chunks
        retrieved_docs = self.retriever.retrieve(query, top_k=3)

        # 2. Find the first valid structured chunk
        structured_doc = None
        for doc in retrieved_docs:
            if doc.get('document') and doc['document'].metadata.get("has_structure", False):
                structured_doc = doc
                break
        
        if not structured_doc:
            return {
                "answer": "I couldn't find any structured data (tables) relevant to your query.",
                "sources": [], "latency": time.time() - start_time, "pipeline": "structured",
            }

        try:
            # 3. Load the DataFrame from the structured JSON in metadata
            table_json = structured_doc['document'].metadata["structured_data_json"]
            df = pd.read_json(table_json, orient="split")
            
            df_schema = df.dtypes.to_string()
            df_head = df.head().to_string()

            # 4. Generate pandas code from LLM
            prompt = self._generate_pandas_code_prompt(query, df_schema, df_head)
            generated_code = (await self.llm_client.generate_async(
                model=query_analysis['strategy'].model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=100
            )).strip()
            
            logger.info(f"Generated pandas code: {generated_code}")

            # 5. Execute the code
            execution_result = self._execute_pandas_code(generated_code, df)

            # 6. Format the result from the code execution to be used as context
            if isinstance(execution_result, (pd.DataFrame, pd.Series)):
                code_output_str = execution_result.to_string()
            else:
                code_output_str = str(execution_result)
            
            logger.info(f"Pandas code output:\n{code_output_str}")

            # 7. Generate a structured, natural language answer based on the code's output
            # This is the second LLM call, focused on presentation.
            final_answer_context = f"The analysis of the structured data returned the following result:\n\n---\n{code_output_str}\n---"
            
            answer = await self.llm_client.generate_structured_response(
                context=final_answer_context,
                query=query,
                model=query_analysis['strategy'].model
            )

        except Exception as e:
            logger.error(f"Error during StructuredPipeline execution: {e}")
            return {
                "answer": f"I encountered an error while analyzing the structured data: {e}",
                "sources": [], "latency": time.time() - start_time, "pipeline": "structured",
            }

        sources = []
        if structured_doc:
            metadata = structured_doc['document'].metadata
            sources.append({
                "file": metadata.get("file_name", "Unknown"),
                "chunk_id": metadata.get("chunk_id", "Unknown"),
                "vector_score": float(structured_doc.get("retrieval_score", 0.0)),
                "content_type": "table"
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "latency": time.time() - start_time,
            "query_type": query_analysis.get("query_type", "structured_query"),
            "pipeline": "structured",
        }

    def stream_answer(self, context: str, query: str, strategy: Dict[str, Any], metadata_response: Dict[str, Any]):
        # Streaming is less applicable to code-generation pipelines, but we can stream the final result.
        import json
        yield f"data: {json.dumps({'type': 'metadata', 'data': metadata_response})}\n\n"
        
        answer = metadata_response.get("answer", "No answer generated.")
        token_event = {"type": "token", "data": answer}
        yield f"data: {json.dumps(token_event)}\n\n"