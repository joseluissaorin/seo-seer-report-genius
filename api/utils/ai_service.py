import logging
import json
import requests
from typing import Dict, Any, Literal, Optional
import re
import time # Keep time for potential delays/retries if needed later

# Potentially use official SDKs if preferred and installed
# from openai import OpenAI, AsyncOpenAI
# from google.generativeai import GenerativeModel
# from anthropic import Anthropic, AsyncAnthropic

# Define Model Types
ModelType = Literal["chatgpt", "gemini", "anthropic", "perplexity"]

# --- Configuration (Consider moving to a config file or env vars) ---
# Placeholder URLs/Models - Adjust as needed
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4" # Or specific model like gpt-4-turbo
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-3-opus-20240229"
ANTHROPIC_API_VERSION = "2023-06-01"
GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
GEMINI_MODEL = "gemini-1.5-flash-latest" # Or gemini-1.5-pro-latest
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = "sonar-pro"

class AIService:
    """Handles interactions with various AI model APIs."""

    def __init__(self, timeout: int = 90):
        self.timeout = timeout
        logging.info("AIService initialized.")

    def _make_request(self, url: str, method: str = "POST", headers: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Generic method to make HTTP requests."""
        headers = headers or {}
        json_data = json.dumps(data) if data else None
        logging.debug(f"Making {method} request to {url} with headers: {headers} and data: {json_data}")
        try:
            response = requests.request(method, url, headers=headers, data=json_data, timeout=self.timeout)
            # Log status for debugging
            logging.debug(f"Received status code {response.status_code} from {url}")
            response.raise_for_status()
            # Check for empty response before decoding JSON
            if not response.content:
                logging.warning(f"Received empty response from {url}")
                return {} # Return empty dict for empty response
            return response.json()
        except requests.exceptions.Timeout:
            logging.error(f"Request timed out after {self.timeout} seconds for URL: {url}")
            raise TimeoutError(f"API request timed out to {url}.")
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP Error {e.response.status_code} for URL {url}: {e.response.text}")
            # Try to parse error details if JSON
            try:
                error_details = e.response.json()
                error_message = error_details.get('error', {}).get('message', e.response.text)
            except json.JSONDecodeError:
                error_message = e.response.text
            raise ConnectionError(f"API request failed ({e.response.status_code}): {error_message}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request Exception for URL {url}: {e}")
            raise ConnectionError(f"API request failed: {e}")
        except json.JSONDecodeError as e:
            response_text = "<Could not read response text>"
            try:
                response_text = response.text
            except Exception as read_err:
                 logging.error(f"Could not read response text from {url}: {read_err}")
            logging.error(f"Failed to decode JSON response from {url}: {e}")
            logging.error(f"Response text snippet: {response_text[:500]}...")
            raise ValueError(f"Invalid JSON received from API at {url}.")

    async def generate_structured_response(
        self,
        prompt: str,
        api_key: str,
        model: ModelType,
        response_format: Literal['json_object', 'text'] = 'json_object',
        system_prompt: Optional[str] = None,
        max_output_tokens: int = 4096 # Default max tokens
    ) -> Dict[str, Any]:
        """
        Generates a structured response (usually JSON) from the selected AI model.
        Uses synchronous `requests` but keeps `async` signature for compatibility.
        """
        logging.info(f"Generating structured response using model: {model}")
        if not api_key or not isinstance(api_key, str) or len(api_key) < 10: # Basic key check
            raise ValueError(f"Valid API key is required for model {model}.")

        raw_result = ""
        try:
            if model == "chatgpt":
                raw_result = self._call_openai(prompt, api_key, response_format, system_prompt, max_output_tokens)
            elif model == "gemini":
                raw_result = self._call_gemini(prompt, api_key, system_prompt, response_format, max_output_tokens)
            elif model == "anthropic":
                raw_result = self._call_anthropic(prompt, api_key, system_prompt, max_output_tokens)
            elif model == "perplexity":
                raw_result = self._call_perplexity(prompt, api_key, system_prompt, max_output_tokens)
            else:
                # This should be caught by Literal type hint, but defensive check
                raise ValueError(f"Unsupported model type: {model}") 

            # Attempt to parse if expecting JSON, otherwise return raw text in a dict
            if response_format == 'json_object':
                try:
                    cleaned_json = self._clean_json_string(raw_result)
                    if not cleaned_json:
                        raise ValueError("Result after cleaning was empty.")
                    parsed_result = json.loads(cleaned_json)
                    # Ensure the top level is a dictionary or list (common valid JSON root types)
                    if not isinstance(parsed_result, (dict, list)):
                        logging.error(f"Parsed JSON is not a dictionary or list for model {model}. Type: {type(parsed_result)}")
                        raise ValueError("Parsed JSON is not a dictionary or list.")
                    # If it was a list, wrap it in a standard dict key for consistency maybe?
                    # Or handle list responses where they are expected.
                    # For now, just return if dict or list
                    if isinstance(parsed_result, dict):
                        return parsed_result
                    else: # It's a list
                        return {"result_list": parsed_result} # Example wrapper
                        
                except (json.JSONDecodeError, ValueError) as e:
                    logging.error(f"Failed to parse JSON response from {model}: {e}")
                    logging.error(f"Cleaned JSON string snippet: {cleaned_json[:500]}...")
                    logging.error(f"Raw response snippet: {raw_result[:500]}...")
                    raise ValueError(f"AI model ({model}) did not return valid JSON after cleaning. Response: {raw_result[:200]}...")
            else:
                 # Return raw text content if text format was expected
                 return {"text_content": raw_result}

        except (TimeoutError, ConnectionError, ValueError) as e:
            logging.error(f"Error during AI call to {model}: {e}")
            raise # Re-raise the specific error
        except Exception as e:
            logging.exception(f"Unexpected error during AI call to {model}: {e}") # Use logging.exception for traceback
            raise RuntimeError(f"An unexpected error occurred with the AI service for model {model}.")

    def _clean_json_string(self, raw_string: str) -> str:
        """Attempts to clean common issues in AI-generated JSON strings."""
        if not isinstance(raw_string, str):
            logging.warning("Input to _clean_json_string was not a string.")
            return ""

        # Remove markdown code fences and optional language identifier more robustly
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_string.strip(), flags=re.MULTILINE).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        # Try to find the start and end of the main JSON object or array
        # Handles cases where there might be leading/trailing text
        brace_start = cleaned.find('{')
        bracket_start = cleaned.find('[')
        
        start_index = -1
        if brace_start != -1 and bracket_start != -1:
            start_index = min(brace_start, bracket_start)
        elif brace_start != -1:
            start_index = brace_start
        elif bracket_start != -1:
            start_index = bracket_start

        if start_index == -1:
            logging.warning("Could not find starting brace or bracket '{' or '[' in the string.")
            return "" # Return empty string if no JSON structure start found
            
        # Find the matching closing brace/bracket requires careful balancing
        # Simple `rfind` might be incorrect for nested structures.
        # A simplified approach for now: find the last brace or bracket
        # More robust parsing might be needed for complex cases.
        last_brace = cleaned.rfind('}')
        last_bracket = cleaned.rfind(']')
        end_index = max(last_brace, last_bracket)

        if end_index == -1 or end_index < start_index:
            logging.warning("Could not find ending brace or bracket '}' or ']' after start.")
            return "" # Return empty string if no valid end found
        
        # Extract the potential JSON part
        potential_json = cleaned[start_index : end_index + 1]
        
        # Basic validation attempt before returning
        try:
            json.loads(potential_json) # Try parsing it
            return potential_json
        except json.JSONDecodeError as e:
            logging.warning(f"Extracted string is still not valid JSON: {e}. Snippet: {potential_json[:200]}...")
            # Fallback: Maybe return the original cleaned string if parsing fails? 
            # Or return empty? Returning empty seems safer.
            return ""

    def _call_openai(self, prompt: str, api_key: str, response_format: str, system_prompt: Optional[str], max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": 0.5, # Lower temperature for predictable JSON
            "max_tokens": max_tokens
        }
        if response_format == 'json_object':
            # Ensure the model supports JSON mode (check OpenAI docs for specific model)
            # gpt-4-turbo and later models support this reliably
            data["response_format"] = { "type": "json_object" }

        response_data = self._make_request(OPENAI_API_URL, headers=headers, data=data)
        # Validate response structure
        if not response_data.get('choices') or not isinstance(response_data['choices'], list) or len(response_data['choices']) == 0:
            raise ValueError("Invalid response from OpenAI API: Missing or invalid 'choices' array.")
        message = response_data['choices'][0].get('message')
        if not message or not isinstance(message, dict) or 'content' not in message:
             raise ValueError("Invalid response from OpenAI API: Missing or invalid 'message' or 'content'.")
        return message['content'] or "" # Return empty string if content is None

    def _call_gemini(self, prompt: str, api_key: str, system_prompt: Optional[str], response_format: str, max_tokens: int) -> str:
        url = GEMINI_API_URL_TEMPLATE.format(model=GEMINI_MODEL, api_key=api_key)
        headers = {"Content-Type": "application/json"}
        contents = []
        
        # Gemini 1.5 supports system instruction
        system_instruction = None
        if system_prompt:
            system_instruction = {"role": "system", "parts": [{"text": system_prompt}]}
        
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": max_tokens, # Use passed max_tokens
            }
        }
        # Add system instruction if provided and model likely supports it
        if system_instruction:
            data["system_instruction"] = system_instruction
            
        # Request JSON output if format is json_object
        if response_format == 'json_object':
             data["generationConfig"]["responseMimeType"] = "application/json"

        response_data = self._make_request(url, headers=headers, data=data)
        # Check for presence of 'candidates'
        if not response_data.get('candidates') or not isinstance(response_data['candidates'], list) or len(response_data['candidates']) == 0:
            prompt_feedback = response_data.get('promptFeedback')
            if prompt_feedback and prompt_feedback.get('blockReason'):
                 block_reason = prompt_feedback.get('blockReason')
                 safety_ratings = prompt_feedback.get('safetyRatings', [])
                 logging.error(f"Gemini request blocked. Reason: {block_reason}, Ratings: {safety_ratings}")
                 raise ValueError(f"Gemini request blocked ({block_reason}). Check content safety guidelines.")
            raise ValueError("Invalid response from Gemini API: Missing or invalid 'candidates' array.")

        candidate = response_data['candidates'][0]
        # Check finish reason first
        finish_reason = candidate.get('finishReason')
        if finish_reason and finish_reason not in ["STOP", "MAX_TOKENS"]:
            # Other reasons like SAFETY, RECITATION, OTHER are typically errors
            safety_ratings = candidate.get('safetyRatings', [])
            logging.error(f"Gemini generation finished unexpectedly. Reason: {finish_reason}, Safety: {safety_ratings}")
            raise ValueError(f"Gemini generation failed ({finish_reason}). Check content safety or other issues.")
            
        # Now check content structure
        if not candidate.get('content') or not isinstance(candidate['content'], dict) or not candidate['content'].get('parts') or not isinstance(candidate['content']['parts'], list) or len(candidate['content']['parts']) == 0:
            logging.error(f"Invalid response structure from Gemini API: Missing or invalid content parts. Finish reason: {finish_reason}")
            raise ValueError("Invalid response structure from Gemini API: Missing or invalid 'content' or 'parts'.")

        return candidate['content']['parts'][0].get('text', "") # Return text, default to empty string

    def _call_anthropic(self, prompt: str, api_key: str, system_prompt: Optional[str], max_tokens: int) -> str:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json"
        }
        messages = []
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": max_tokens, # Use passed max_tokens
            "messages": messages,
            "temperature": 0.5
        }
        if system_prompt:
            data["system"] = system_prompt

        response_data = self._make_request(ANTHROPIC_API_URL, headers=headers, data=data)
        # Check response structure
        if response_data.get('type') == 'error':
            error_data = response_data.get('error', {})
            error_type = error_data.get('type')
            error_message = error_data.get('message')
            logging.error(f"Anthropic API Error: {error_type} - {error_message}")
            raise ValueError(f"Anthropic API error ({error_type}): {error_message}")
            
        if not response_data.get('content') or not isinstance(response_data['content'], list) or len(response_data['content']) == 0:
             raise ValueError("Invalid response from Anthropic API: Missing or invalid 'content' array.")
             
        # Find the first text block
        first_text_block = next((block for block in response_data['content'] if block.get('type') == 'text'), None)
        
        if not first_text_block or 'text' not in first_text_block:
             raise ValueError("No valid text content block found in Anthropic response.")
             
        return first_text_block['text'] or "" # Return empty string if text is None

    def _call_perplexity(self, prompt: str, api_key: str, system_prompt: Optional[str], max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": PERPLEXITY_MODEL,
            "messages": messages,
            "max_tokens": min(max_tokens, 8000) # Respect Perplexity limit if needed
            # Add other params like temperature if needed
            # "temperature": 0.7 
        }

        response_data = self._make_request(PERPLEXITY_API_URL, headers=headers, data=data)
        # Check response structure
        if not response_data.get('choices') or not isinstance(response_data['choices'], list) or len(response_data['choices']) == 0:
            raise ValueError("Invalid response from Perplexity API: Missing or invalid 'choices' array.")
        message = response_data['choices'][0].get('message')
        if not message or not isinstance(message, dict) or 'content' not in message:
             raise ValueError("Invalid response from Perplexity API: Missing or invalid 'message' or 'content'.")
        return message['content'] or "" # Return empty string if content is None


# Example Usage (placeholder)
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ai_service = AIService()
    # Replace with actual keys and desired model/prompt
    # IMPORTANT: Load keys securely, e.g., from environment variables
    import os
    api_key_placeholder = os.getenv("PERPLEXITY_API_KEY", "YOUR_PPLX_KEY_HERE") # Example for Perplexity
    # api_key_placeholder = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
    # api_key_placeholder = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_KEY_HERE")
    # api_key_placeholder = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_KEY_HERE")
    
    model: ModelType = "perplexity" # or chatgpt, gemini, anthropic
    test_prompt = "Explain the concept of SEO in simple terms. Respond ONLY with a JSON object like: {\"explanation\": \"...\"}"
    test_system_prompt = "You are a helpful SEO assistant designed to output JSON."

    if api_key_placeholder.startswith("YOUR_"):
        print("\n--- WARNING: Using placeholder API Key. AI call will likely fail. ---")
        print("--- Please set the corresponding environment variable (e.g., PERPLEXITY_API_KEY) ---")

    try:
        print(f"Testing model: {model}")
        result = await ai_service.generate_structured_response(
            prompt=test_prompt,
            api_key=api_key_placeholder,
            model=model,
            response_format='json_object', # or 'text'
            system_prompt=test_system_prompt
        )
        print("\n--- AI Service Result ---")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\n--- AI Service Error ---")
        print(f"Error: {e}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
 