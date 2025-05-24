from Models.llm_base_model import LLMBaseModel
import tiktoken
import base64
import io
from PIL import Image
import logging
import os
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel
from langfuse.openai import AzureOpenAI
from openai._types import NOT_GIVEN, NotGiven
from pydantic import BaseModel
from gen_utils.parsing_utils import load_config, initialize_environment_variables
import json 
import tiktoken

class GPTModel(LLMBaseModel):
    """
    GPT-based model using Azure OpenAI, inherits from LLMBaseModel
    and overrides token-based chunking and image transformations.
    
    Implements:
      - `_do_single_inference(...)` for the Azure OpenAI API call
      - `count_tokens(...)` using tiktoken
      - `chunk_content(...)` for token-based text chunking
      - `_transform_image(...)` converting images to base64 dictionaries
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str,Any]] = None,
        **kwargs
    ):
        """
        Initialize GPTModel, reading config for Azure OpenAI details and setting up 
        tiktoken for token counting.

        Args:
            logger (Optional[logging.Logger]): A logger for debug/info messages. If not provided, a default is created.
            config (Optional[Dict[str, Any]]): A dictionary of configuration parameters 
                loaded from JSON (e.g. keys like 'secret_name', 'project_id', 'base_temperature', etc.).
                If not provided, we load from './configs/config.json'.
            **kwargs: Additional arguments passed to LLMBaseModel or used internally.
        """
        super().__init__(logger=logger, **kwargs)

        if not config:
            config = load_config("./configs/config.json")

        self.logger = logger or logging.getLogger(__name__)
        self.config = config

        initialize_environment_variables(
            secret_name=config['secret_name'],
            project_id=config['project_id'],
            logger=self.logger
        )

        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-08-01-preview"
        )

        self.model_name = config['llm-model-deployment-name']
        self.base_temperature = config.get('base_temperature', 0.2)
        self.max_chunk_tokens = config.get('max_chunk_size', 2000)

        # Attempt to load a token encoder
        try:
            self.encoder = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            self.logger.error(f"Model {self.model_name} not found in tiktoken. Falling back to o200k_base.")
            self.encoder = tiktoken.get_encoding("o200k_base")

    def _do_single_inference(
        self,
        content_parts: List[Union[str, Dict[str, Dict[str,str]]]],
        response_format: Union[Type[BaseModel], BaseModel, None],
        response_format_flag: bool,
        **kwargs
    ) -> Union[Type[BaseModel], BaseModel, str]:
        """
        Performs a single call to AzureOpenAI, constructing 'messages' from content_parts.
        If response_format_flag=True and response_format is a pydantic model, 
        we parse structured JSON. Otherwise, we return raw text.

        Args:
            content_parts (List[Union[str, Dict[str, Dict[str,str]]]]): 
                Typically a list with 1 or more items, 
                e.g. [system_text, user_text] or [system_text, image_dict], ...
            response_format (Union[Type[BaseModel], BaseModel, None]): 
                A pydantic model for structured output, or None for raw text.
            response_format_flag (bool): 
                True => we attempt structured parse with .parse(...) 
                from the AzureOpenAI beta chat completions.
            **kwargs: Additional arguments, e.g. 'temperature'.

        Returns:
            Any: 
                - A dict if structured parse succeeded, 
                - or a string for raw text. 
        """
        temperature = kwargs.get("temperature", self.base_temperature)

        # Build messages from content_parts
        messages = []
        if len(content_parts) == 1:
            # Single user message
            messages.append({"role": "user", "content": str(content_parts[0])})
        else:
            # The first item is typically the "system" instruction
            messages.append({"role": "system", "content": str(content_parts[0])})
            for i in range(1, len(content_parts)):
                # If it's a string, treat as user. If it's something else, we might handle differently
                if isinstance(content_parts[i], str):
                    user_msg = str(content_parts[i])
                else:
                    # Potentially an image dict or list
                    user_msg = [content_parts[i]]  # Convert to string or do partial approach
                messages.append({"role": "user", "content": user_msg})

        if response_format_flag and response_format:
            structured_resp = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=response_format,
                timeout=60,
                temperature=temperature,
                user_id="gpt-parser"
            )
            self.logger.debug(f"GPT structured response: {structured_resp.choices[0].message.parsed}")
            return structured_resp.choices[0].message.parsed
        else:
            raw_resp = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                user_id="gpt-parser"
            )
            self.logger.debug(f"GPT raw response: {raw_resp.choices[0].message.content}")
            return raw_resp.choices[0].message.content

    def count_tokens(self, content: List[str]) -> int:
        """
        Measures the approximate number of tokens in 'content' using tiktoken.
        If content is a list of strings (images are processed one-by-one), we sum the tokens of each item.

        Args:
            content (Any): The text or list of text items to measure tokens for.

        Returns:
            int: The token count for the input content.
        """
        if isinstance(content, list):
            total = 0
            for c in content:
                total += len(self.encoder.encode(str(c)))
            return total
        else:
            return len(self.encoder.encode(str(content)))

    def chunk_content(self, text: str, max_chunk_tokens: int, **kwargs) -> List[str]:
        """
        Token-based chunking using tiktoken. We break 'text' into 
        segments each up to 'max_chunk_tokens' tokens.

        Args:
            text (str): The text to be chunked.
            max_chunk_tokens (int): Max tokens per chunk.

        Returns:
            List[str]: A list of text chunks. 
        """
        all_tokens = self.encoder.encode(text)
        chunks: List[str] = []
        start = 0
        while start < len(all_tokens):
            end = start + max_chunk_tokens
            chunk = all_tokens[start:end]
            chunk_str = self.encoder.decode(chunk)
            chunks.append(chunk_str)
            start = end
        return chunks

    def _transform_image(
        self,
        img_path: str,
        system_instruction: str = "",
        **kwargs
    ) -> List[Union[str, Dict[str, Dict[str,str]]]]:
        """
        Converts an image file to a base64-encoded dictionary for GPT.
        If processing fails, the exception is raised so that the caller (_handle_vision_inputs)
        can skip the problematic image.

        Args:
            img_path (str): The filesystem path to the image file.
            system_instruction (str): Optional system text to prepend.
            **kwargs: Additional arguments if needed.

        Returns:
            List[Union[str, Dict[str, Dict[str, str]]]]:
                A list containing an optional system instruction (str) and a dictionary
                representing the image in "image_url" format.
        
        Raises:
            Exception: Propagates any exception encountered while reading or processing the image.
        """

        try:
            with Image.open(img_path) as img:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                base64_img = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Failed to transform image {img_path} for GPT: {e}")
            raise e

        content_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_img}",
                "detail": "high"
            }
        }
        parts: List[Any] = []
        if system_instruction:
            parts.append(system_instruction)
        parts.append(content_image)
        return parts