import os
import logging
import time
import random
import json
from typing import Any, Dict, List, Optional, Union, Type
from langfuse.decorators import langfuse_context, observe
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image, GenerationConfig
from pydantic import BaseModel
from gen_utils.parsing_utils import load_config, initialize_environment_variables

from Models.llm_base_model import LLMBaseModel

class GeminiModel(LLMBaseModel):
    """
    Gemini model using Google Vertex AI generative models.
    Inherits from LLMBaseModel for shared logic (retries, merges, chunking).
    Implements:
      - _do_single_inference: calls self.model.generate_content
      - count_tokens: uses model.count_tokens(...) if available, else fallback
      - _transform_image: produce Part.from_image(...) objects
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Initializes the GeminiModel with Vertex AI generative model.

        Environment variables checked:
          - GEMINI_PROJECT_ID (default: "your-project-id")
          - GEMINI_LOCATION   (default: "us-central1")
          - GEMINI_MODEL      (default: "gemini-2.0-flash-001")

        Args:
            logger (Optional[logging.Logger]): A logger instance for debugging.
            **kwargs: Additional arguments to pass to the parent constructor or store.
        """
        # Call base constructor
        super().__init__(logger=logger, **kwargs)

        # Load config from environment or fallback
        self.project_id = os.getenv("GEMINI_PROJECT_ID", "your-project-id")

        logger = logging.getLogger(__name__)

        # Load config JSON
        config = load_config("./configs/config.json")

        # Retrieve GCP secret or load dotenv and store in environment
        initialize_environment_variables(secret_name=config['secret_name'],project_id=config['project_id'],logger=logger)

        # Load GCP config from environment variables (or provide defaults).
        self.project_id = os.getenv("GEMINI_PROJECT_ID", "cd-ds-384118")
        self.region = os.getenv("GEMINI_LOCATION", "us-central1")
        self.model_id = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
        self.logger = logger 

        # Initialize Vertex AI with project/region
        vertexai.init(project=self.project_id, location=self.region)

        # Create the generative model
        self.model = GenerativeModel(model_name=self.model_id)

    def _do_single_inference(
        self,
        content_parts: List[Union[str, Part]],
        response_format: Union[Type[BaseModel], BaseModel, None],
        response_format_flag: bool,
        max_retries: int = 3,
        jitter: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], str]:
        """
        Attempts a Gemini call up to `max_retries` times with exponential backoff
        and optional jitter. Wraps `_call_gemini_once`.
        
        Args:
            content_parts: A list of either strings or Part objects (images, text).
            response_format: A pydantic class or instance if structured JSON is desired,
                             else None if raw text is fine.
            response_format_flag: Whether to request structured JSON from Gemini.
            max_retries: Maximum number of attempts if transient errors occur.
            jitter: If True, adds random jitter to the exponential backoff wait time.

        Returns:
            Either a dict (JSON) if the parse was successful or raw text (str).
        
        Raises:
            RuntimeError: If all retry attempts fail.
            Exception: Propagates the last exception encountered if max_retries exhausted.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                return self._call_gemini_once(
                    content_parts=content_parts,
                    response_format=response_format,
                    response_format_flag=response_format_flag
                )
            except TimeoutError:
                attempt += 1
                self.logger.warning(
                    f"TimeoutError: Attempt {attempt}/{max_retries} for Gemini. Retrying..."
                )
                wait_seconds = (2 ** attempt) * (1 + random.random()) if jitter else 2 ** attempt
                time.sleep(wait_seconds)
            except Exception as e:
                attempt += 1
                self.logger.error(f"Gemini call error on attempt {attempt}: {e}")
                if attempt >= max_retries:
                    raise e
                wait_seconds = (2 ** attempt) * (1 + random.random()) if jitter else 2 ** attempt
                time.sleep(wait_seconds)

        raise RuntimeError("Gemini request failed after maximum retries.")

    # -------------------------------------------------------------------------
    # ============== SINGLE GEMINI CALL (No Retry) ==============
    # -------------------------------------------------------------------------
    @observe(as_type="generation")
    def _call_gemini_once(
        self,
        content_parts: List[Union[str, Part]],
        response_format: Union[Type[BaseModel], BaseModel, None],
        response_format_flag: bool
    ) -> Union[Dict[str, Any], str]:
        """
        Single call to Gemini. If response_format_flag=True, attempts to build a 
        GenerationConfig with response_schema from the given pydantic model.

        Args:
            content_parts (List[Union[str, Part]]): 
                The message content, possibly including strings (system/user text) 
                and Part objects (for images).
            response_format (Union[Type[BaseModel], BaseModel, None]): 
                If a pydantic model, we can pass it in as a schema. Otherwise None.
            response_format_flag (bool): 
                If True, we attempt structured JSON output by sending response_schema 
                in GenerationConfig.
            **kwargs: Additional arguments (e.g., temperature) if needed.

        Returns:
            Union[Dict[str, Any], str]: 
                - A Python dict if we manage to parse JSON output, 
                  or if we fallback to json.loads(response.text).
                - A raw string if we do a simple text call with no schema config.
        """
        # Possibly handle temperature or other generation config in **kwargs
        if response_format_flag and response_format is not None:
            # Convert the pydantic model into a JSON schema
            schema_dict = {}
            if hasattr(response_format, "model_json_schema"):
                schema_dict = response_format.model_json_schema()

            gen_config = GenerationConfig(
                response_mime_type="application/json",
                response_schema=schema_dict
            )
            response = self.model.generate_content(
                contents=content_parts,
                generation_config=gen_config
            )

            # pass model, model input, and usage metrics to Langfuse
            langfuse_context.update_current_observation(
                input=content_parts,
                model=self.model_id,
                usage_details={
                    "input": response.usage_metadata.prompt_token_count,
                    "output": response.usage_metadata.candidates_token_count,
                    "total": response.usage_metadata.total_token_count
                }
            )
            
            parsed_obj = getattr(response, "parsed", None)
            self.logger.debug(f"[Gemini] Structured response text: {response.text}")
            if parsed_obj is not None:
                return parsed_obj
            else:
                # fallback to raw JSON parse if structured parse is incomplete
                # self.logger.warning("[Gemini] JSON parse failed or partial. Returning raw text as JSON.")
                return json.loads(response.text)
        else:
            # Simple text generation call
            response = self.model.generate_content(contents=content_parts)

            self.logger.debug(f"[Gemini] Raw response text: {response.text}")

            langfuse_context.update_current_observation(
                input=content_parts,
                model=self.model_id,
                usage_details={
                    "input": response.usage_metadata.prompt_token_count,
                    "output": response.usage_metadata.candidates_token_count,
                    "total": response.usage_metadata.total_token_count
                }
            )
            return response.text

    def count_tokens(self, content: List[Union[str, Part]]) -> int:
        """
        Attempts to count tokens using self.model.count_tokens if it exists, 
        else fallback to naive len(str(content)) approach.

        Args:
            content (Any): The content to measure tokens for (e.g. text string or list of strings).

        Returns:
            int: An approximate number of tokens for the given content.
        """
        try:
            token_info = self.model.count_tokens(contents=content)
        except AttributeError:
            # If model.count_tokens isn't supported or results in error
            self.logger.warning("[Gemini] .count_tokens not available, using naive length.")
            return len(str(content))
        except Exception as e:
            self.logger.warning(f"[Gemini] count_tokens call error: {e}. Using naive length.")
            return len(str(content))

    def _transform_image(
        self,
        system_instruction: str = "",
        img_path: str="",
        **kwargs
    ) -> List[Union[str, Part]]:
        """
        Converts an image file path into a Part.from_image(...) for Gemini.
        If processing fails, the exception is raised so that the caller (_handle_vision_inputs)
        can skip the problematic image.

        Args:
            img_path (str): Path to the image file on disk.
            system_instruction (str, optional): Optional system text to prepend.
            **kwargs: Additional arguments if needed.

        Returns:
            List[Union[str, Part]]:
                A list typically containing an optional system instruction (str) and 
                a Part object representing the image.
        
        Raises:
            Exception: Propagates any exception encountered while loading or processing the image.
        """
        try:
            loaded_img = Image.load_from_file(img_path)
            image_part = Part.from_image(loaded_img)
        except Exception as e:
            self.logger.error(f"Failed to transform image {img_path} for Gemini: {e}")
            raise e

        result: List[Union[str, Part]] = []
        if system_instruction:
            result.append(system_instruction)
        result.append(image_part)
        return result

    # # -------------------------------------------------------------------------
    # # ============== MASTER generate_response ==============
    # # -------------------------------------------------------------------------
    # def generate_response(
    #     self,
    #     system_instruction: Optional[str] = None,
    #     user_instruction: Union[str, List[str], None] = None,
    #     response_format: Union[Type[BaseModel], BaseModel, None] = None,
    #     response_format_flag: bool = False,
    #     merge_system_prompt: Optional[str] = None,
    #     max_retries: int = 3,
    #     jitter: bool = True,
    #     img_path: str,
    #     system_instruction: str = "",
    #     **kwargs
    # ) -> List[Union[str, Part]]:
    #     """
    #     Converts an image file path into a Part.from_image(...) for Gemini.
    #     If processing fails, the exception is raised so that the caller (_handle_vision_inputs)
    #     can skip the problematic image.

    #     Args:
    #         img_path (str): Path to the image file on disk.
    #         system_instruction (str, optional): Optional system text to prepend.
    #         **kwargs: Additional arguments if needed.

    #     Returns:
    #         List[Union[str, Part]]:
    #             A list typically containing an optional system instruction (str) and 
    #             a Part object representing the image.
        
    #     Raises:
    #         Exception: Propagates any exception encountered while loading or processing the image.
    #     """
    #     try:
    #         loaded_img = Image.load_from_file(img_path)
    #         image_part = Part.from_image(loaded_img)
    #     except Exception as e:
    #         self.logger.error(f"Failed to transform image {img_path} for Gemini: {e}")
    #         raise e

    #     result: List[Union[str, Part]] = []
    #     if system_instruction:
    #         result.append(system_instruction)
    #     result.append(image_part)
    #     return result
