import ollama
from src.core.logging_config import logger
from typing import List, Dict, Any, Tuple
from src.config.settings import settings
import json
import re
class OllamaClient:
    """
    A client for interacting with the Ollama API, designed to be robust and provide
    observability by tracking token usage.
    """

    async def _generate(
        self, model: str, messages: List[Dict[str, Any]], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Private method to handle the core logic of making a request to the Ollama API.
        It centralizes the API call, error handling, and response parsing.

        Returns:
            A dictionary containing 'content' and 'token_usage'.
        """
        try:
            client = ollama.AsyncClient()
            response = await client.chat(
                model=model, messages=messages, options=options
            )
            
            token_usage = {
                "input_tokens": response.get("prompt_eval_count", 0),
                "output_tokens": response.get("eval_count", 0),
            }
            
            logger.info(
                f"Ollama generation successful. Input tokens: {token_usage['input_tokens']}, "
                f"Output tokens: {token_usage['output_tokens']}"
            )

            return {
                "content": response.get("message", {}).get("content", ""),
                "token_usage": token_usage,
            }
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}", exc_info=True)
            # Attempt to provide a more specific error if the model is not found
            try:
                ollama.show(model)
            except Exception as show_error:
                logger.error(f"Model '{model}' may not be available. Error: {show_error}")
                error_message = f"Error: Model '{model}' not found. Please ensure it is installed and available."
            else:
                error_message = f"Error: Could not get a response from the model. Details: {e}"
            
            return {
                "content": error_message,
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
            }

    async def generate_structured_response(
        self, context: str, query: str, model: str = settings.DEFAULT_LLM_MODEL
    ) -> Dict[str, Any]:
        """
        Generates a structured response using a detailed prompt template.
        Returns a dictionary with 'content' and 'token_usage'.
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

    async def generate_with_system_prompt(
        self, system_prompt: str, user_prompt: str, model: str = settings.DEFAULT_LLM_MODEL
    ) -> Dict[str, Any]:
        """
        Generates a response using both a system and a user prompt.
        Returns a dictionary with 'content' and 'token_usage'.
        """
        logger.info(f"Generating response with system prompt using model {model}.")
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        options = {
            "num_predict": 1024,
            "temperature": 0.1,
        }
        return await self._generate(model=model, messages=messages, options=options)

    async def generate_from_prompt(
        self, prompt: str, model: str = settings.DEFAULT_LLM_MODEL
    ) -> Dict[str, Any]:
        """
        Generates a response from a direct prompt string.
        Returns a dictionary with 'content' and 'token_usage'.
        """
        messages = [{'role': 'user', 'content': prompt}]
        options = {"num_predict": 256, "temperature": 0.2, "top_p": 0.9}
        return await self._generate(model=model, messages=messages, options=options)

    async def generate_with_image(
        self, 
        model: str, 
        prompt: str, 
        system_prompt: str, 
        images_base64: List[str]
    ) -> Dict[str, Any]:
        """
        Generates a response from a multimodal model using a prompt and images.
        Returns a dictionary with 'content' and 'token_usage'.
        """
        logger.info(f"Generating response with {len(images_base64)} image(s) using model {model}.")
        messages = [
            {'role': 'system', 'content': system_prompt},
            {
                'role': 'user',
                'content': prompt,
                'images': images_base64,
            }
        ]
        options = {
            "num_predict": 1024,
            "temperature": 0.1,
        }
        return await self._generate(model=model, messages=messages, options=options)



    def clean_json_response(self, response_text: str) -> str:
        """
        Cleans the raw text response from an LLM to extract a valid JSON object or array.
        This function is designed to be robust against conversational text surrounding the JSON.
        """
        # Attempt to find a JSON object or array using a regular expression
        # This regex looks for a string starting with '{' or '[' and ending with '}' or ']'
        # It handles nested structures.
        match = re.search(r'(\[.*?\].*?|\{.*?\}.*?)', response_text, re.DOTALL)

        if not match:
            logger.error("Could not find a JSON object or array in the response.")
            raise ValueError("No JSON object or array found in response")

        json_str = match.group(0)

        # Further clean up to remove potential markdown code blocks
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]

        # Validate that the extracted string is valid JSON
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            logger.error(f"Extracted string is not valid JSON: {json_str}", exc_info=True)
            raise ValueError(f"Extracted string could not be parsed as JSON: {e}")