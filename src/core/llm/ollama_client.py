import ollama
import logging
from typing import List, Dict, Any
from src.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class OllamaClient:
    """
    A client for interacting with the Ollama API using the official 'ollama' library.
    """

    async def generate_structured_response(self, context: str, query: str, model: str = settings.DEFAULT_LLM_MODEL) -> str:
        """
        Generates a structured response using a detailed prompt template.
        """
        system_prompt = (
            "**System Prompt: You are a senior AI systems engineer.**\n\n"
            "Your task is to provide a clear, concise, and structured answer to the user's query, "
            "basing your response *exclusively* on the context provided below. Do not use any external knowledge."
        )
        
        user_prompt = (
            "**Instructions:**\n"
            "1.  **Analyze the Context:** Carefully review all the provided context documents.\n"
            "2.  **Synthesize the Answer:** Formulate a direct answer to the user's query based on the information in the context.\n"
            "3.  **Cite Sources:** For each piece of information you use, you MUST cite the corresponding context document using the format `[Source X]`, where 'X' is the context number.\n"
            "4.  **Structure the Output:** Format your response into two sections:\n"
            "    *   **Direct Answer:** A concise, immediate answer to the user's question.\n"
            "    *   **Detailed Explanation:** A more thorough explanation, elaborating on the answer and synthesizing information from multiple sources. Ensure all claims are supported by citations.\n"
            "5.  **Handle Missing Information:** If the context does not contain the information needed to answer the query, state clearly: 'The provided context does not contain enough information to answer this question.' Do not attempt to answer.\n\n"
            "--- CONTEXT ---\n"
            "{context}\n"
            "--- END CONTEXT ---\n\n"
            "**User Query:** {query}\n\n"
            "**Your Response:**"
        ).format(context=context, query=query)

        return await self.generate_with_system_prompt(system_prompt, user_prompt, model)

    async def generate_response(self, context: str, query: str, model: str = settings.DEFAULT_LLM_MODEL) -> str:
        """
        Generates a response using the Ollama API.

        Args:
            context (str): The formatted context string.
            query (str): The user's query.
            model (str): The name of the model to use (e.g., "phi3", "llama3").

        Returns:
            str: The generated response from the model.
        """
        prompt = f"""You are an AI assistant answering questions using the provided context.

Instructions:
- Use ONLY information from the context.
- If the answer cannot be directly supported by the context, explicitly state: "The answer is not available in the provided documents."
- Do not infer or guess.
- Be concise (max 6 sentences).
- Do not repeat the question.
- Do not add extra explanations.

Context:
---
{context}
---

Question:
{query}

Answer:
"""
        
        logging.info(f"Prompt length: {len(prompt)} characters")

        try:
            client = ollama.AsyncClient()
            response = await client.chat(
                model=model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    },
                ],
                options={
                    "num_predict": 512,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                }
            )
            
            generated_text = response['message']['content']
            logging.info(f"Generated response length: {len(generated_text)} characters")
            return generated_text

        except Exception as e:
            logging.error(f"Error calling Ollama API: {e}")
            # Check if the model exists
            try:
                ollama.show(model)
            except Exception as show_error:
                logging.error(f"Model '{model}' may not be available. Error: {show_error}")
                return f"Error: Model '{model}' not found. Please ensure it is installed and available."
            return f"Error: Could not get a response from the model. Details: {e}"

    def clean_json_response(self, response_text: str) -> str:
        """
        Cleans the raw text response from an LLM to extract a valid JSON object.
        """
        # Find the start and end of the JSON object
        start_brace = response_text.find('{')
        end_brace = response_text.rfind('}') + 1
        
        if start_brace == -1 or end_brace == 0:
            logger.error("Could not find a JSON object in the response.")
            raise ValueError("No JSON object found in response")
            
        json_str = response_text[start_brace:end_brace]
        return json_str

    async def generate_with_image(
        self, 
        model: str, 
        prompt: str, 
        system_prompt: str, 
        images_base64: List[str]
    ) -> str:
        """
        Generates a response from a multimodal model using a prompt and a list of base64-encoded images.
        """
        logger.info(f"Generating response with {len(images_base64)} image(s) using model {model}.")
        try:
            client = ollama.AsyncClient()
            
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': prompt,
                    'images': images_base64,
                }
            ]
            
            response = await client.chat(
                model=model,
                messages=messages,
                options={
                    "num_predict": 1024, # Increased for potentially complex JSON output
                    "temperature": 0.1,
                }
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error calling Ollama API with image: {e}", exc_info=True)
            return f"Error: Could not get a response from the multimodal model. Details: {e}"


    async def generate_from_prompt(self, prompt: str, model: str = settings.DEFAULT_LLM_MODEL) -> str:
        """
        Generates a response from a direct prompt string, without context/query formatting.
        """
        try:
            client = ollama.AsyncClient()
            response = await client.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={"num_predict": 256, "temperature": 0.2, "top_p": 0.9}
            )
            return response['message']['content']
        except Exception as e:
            logging.error(f"Error calling Ollama API from prompt: {e}")
            return f"Error: Could not get a response from the model. Details: {e}"

    async def generate_with_system_prompt(self, system_prompt: str, user_prompt: str, model: str = settings.DEFAULT_LLM_MODEL) -> str:
        """
        Generates a response using both a system and a user prompt.
        """
        logger.info(f"Generating response with system prompt using model {model}.")
        try:
            client = ollama.AsyncClient()
            
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            
            response = await client.chat(
                model=model,
                messages=messages,
                options={
                    "num_predict": 1024, # Increased for potentially complex JSON output
                    "temperature": 0.1,
                }
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error calling Ollama API with system prompt: {e}", exc_info=True)
            return f"Error: Could not get a response from the model. Details: {e}"


if __name__ == '__main__':
    # Example Usage
    # NOTE: This requires the Ollama server to be running with the 'phi3' or 'llama3' model.
    # You can run it with: `ollama run phi3`
    try:
        ollama_client = OllamaClient()

        # Simulate a retrieval context
        example_context = """
Context 1:
Source: doc1
Content:
The capital of France is Paris.

Context 2:
Source: doc2
Content:
Paris is known for its art, fashion, and culture.
"""
        user_query = "What is the capital of France and what is it known for?"

        print("Generating response from Ollama...")
        answer = ollama_client.generate_response(context=example_context, query=user_query, model="phi3")
        
        print(f"\nQuery: {user_query}")
        print(f"Answer: {answer}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")