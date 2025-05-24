import logging
import time
import random
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import json 

class LLMBaseModel(ABC):
    """
    Abstract base class for LLM models (GPT-like, Gemini-like, or others).
    Defines common logic for:
      - Retry calls with exponential backoff (_retry_inference)
      - Combining partial responses into a final output (_combine_partial_responses)
      - Processing lists of image paths for vision tasks (_handle_vision_inputs)
      - Chunking text content (chunk_content)
      - Optionally formatting outputs (_format_outputs)

    Child classes must implement:
      - _do_single_inference(...)
      - count_tokens(...)
      - _transform_image(...) if vision usage is expected
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Base constructor for LLMBaseModel.

        Args:
            logger (Optional[logging.Logger]): 
                A logger instance for debug or info messages.
            **kwargs: 
                Additional keyword arguments for potential extension 
                or storing configuration parameters.
        """
        self.logger = logger or logging.getLogger(__name__)

    # -------------------------------------------------------------------------
    # ============== SINGLE CALL/RETRY WRAPPER ==============
    # -------------------------------------------------------------------------

    @abstractmethod
    def _do_single_inference(
        self,
        content_parts: List[Any],
        response_format: Any,
        response_format_flag: bool,
        **kwargs
    ) -> Union[Dict[str, Any], str]:
        """
        Abstract method for a single model API call. Child classes must implement 
        the actual logic to invoke their respective LLM service.

        Args:
            content_parts (List[Any]): 
                A list representing the content to be sent to the LLM, 
                e.g. [system_prompt, user_prompt], or includes vision objects.
            response_format (Any): 
                If structured output is desired (pydantic model, schema, etc.).
            response_format_flag (bool): 
                Indicates whether to use structured parsing logic or raw text approach.
            **kwargs: 
                Additional arguments, e.g. temperature or other model config.

        Returns:
            Union[Dict[str, Any], str]:
                A dict if structured parse is used and succeeds, 
                or raw string content if no structured parse or partial fallback.
        """

    def _retry_inference(
        self,
        content_parts: List[Any],
        response_format: Any,
        response_format_flag: bool,
        max_retries: int = 3,
        jitter: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], str]:
        """
        Shared retry logic with exponential backoff + optional jitter.
        Invokes self._do_single_inference(...) in a try/except loop.

        Args:
            content_parts (List[Any]): 
                Content to be sent to the LLM (text segments, images, etc.).
            response_format (Any): 
                Format or schema for structured output (may be None).
            response_format_flag (bool): 
                True => structured parse attempt, False => raw text.
            max_retries (int): 
                Maximum number of attempts before giving up.
            jitter (bool): 
                Whether to add random jitter to exponential backoff wait times.
            **kwargs:
                Additional arguments forwarded to _do_single_inference.

        Returns:
            Union[Dict[str, Any], str]:
                The final response from _do_single_inference if success,
                or raises an exception after exceeding max_retries.

        Raises:
            RuntimeError: 
                If all retry attempts fail without successful inference.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                return self._do_single_inference(
                    content_parts=content_parts,
                    response_format=response_format,
                    response_format_flag=response_format_flag,
                    **kwargs
                )
            except TimeoutError:
                attempt += 1
                self.logger.warning(f"TimeoutError: Attempt {attempt}/{max_retries}, retrying...")
                wait_seconds = (2 ** attempt) * (1 + random.random()) if jitter else (2 ** attempt)
                time.sleep(wait_seconds)
            except Exception as e:
                # Could be an HTTP or network error
                attempt += 1
                self.logger.error(f"Model call error on attempt {attempt}: {e}")
                if attempt >= max_retries:
                    raise e
                wait_seconds = (2 ** attempt) * (1 + random.random()) if jitter else (2 ** attempt)
                time.sleep(wait_seconds)

        raise RuntimeError("Request failed after maximum retries.")

    # -------------------------------------------------------------------------
    # ============== MERGE PARTIAL RESPONSES ==============
    # -------------------------------------------------------------------------

    def _combine_partial_responses(
        self,
        partials: List[Any],
        base_system_prompt: str,
        merge_system_prompt: str,
        response_format: Any,
        response_format_flag: bool,
        max_retries: int,
        jitter: bool,
        **kwargs
    ) -> Union[Dict[str, Any], str]:
        """
        Merges multiple partial results into one final outcome using a specialized prompt.
        If no merge_system_prompt is provided, we return a dictionary containing the partials.

        Args:
            partials (List[Any]): 
                A list of intermediate results (could be strings or structured dicts).
            base_system_prompt (str): 
                The original or "system" instruction context.
            merge_system_prompt (str):
                If provided, we do a final inference to unify partials. 
                If empty, we return {'chunked_responses': partials}.
            response_format (Any):
                Pydantic model or schema if structured output is desired, else None.
            response_format_flag (bool):
                True => attempt structured parse for the merge result.
            max_retries (int):
                Maximum attempts for the final merge call if transient errors occur.
            jitter (bool):
                Whether to apply random jitter to exponential backoff.
            **kwargs:
                Additional arguments for _retry_inference (e.g. temperature).

        Returns:
            Union[Dict[str, Any], str]:
                The merged final result if merge_system_prompt is used,
                or {'chunked_responses': partials} if no merge prompt is given.
        """
        if not merge_system_prompt:
            return {"chunked_responses": partials}

        final_prompt = f"{base_system_prompt}\n{merge_system_prompt}\n"
        user_text = "Below are partial responses:\n\n"
        for i, p in enumerate(partials):
            user_text += f"--- PART {i+1} ---\n{p}\n\n"

        content = [final_prompt, user_text]

        # Perform one final inference to merge them
        return self._retry_inference(
            content_parts=content,
            response_format=response_format,
            response_format_flag=response_format_flag,
            max_retries=max_retries,
            jitter=jitter,
            **kwargs
        )

    # -------------------------------------------------------------------------
    # ============== HELPER FUNCTIONS (CHUNKING CONTENT, FORMAT OUTPUTS, IMAGES TRANSFORMATION) ==============
    # -------------------------------------------------------------------------

    @abstractmethod
    def count_tokens(self, content: Any) -> int:
        """
        Abstract method for measuring or approximating the token usage of 'content'.

        Args:
            content (Any): 
                Typically text (str) or a list of text segments.

        Returns:
            int: The approximate or exact token count for the content.
        """

    def chunk_content(
        self, 
        content: Any, 
        max_chunk_tokens: int
    ) -> List[Any]:
        """
        Generic method to chunk content by tokens. By default, it's a no-op 
        (returns [content]). Child classes can override to do advanced logic.

        Args:
            content (Any): The text (or other data) to be chunked.
            max_chunk_tokens (int): The max tokens per chunk, if relevant.

        Returns:
            List[Any]: 
                A list of content chunks. Default returns a single-element list containing content.
        """
        return [content]

    def _format_outputs(self, outputs: Any) -> Any:
        """
        Optional hook to format final outputs. Child classes can override if they want 
        a specific structure. By default, we do:
          - If outputs is a list, we try to call model_dump() on each item if available,
            otherwise store in 'chunked_responses'.
          - If outputs is a dict, return as is.
          - If single item, attempt model_dump or return an empty chunked_responses.

        Args:
            outputs (Any): 
                The raw or partial result from an inference or merges.

        Returns:
            Any:
                Typically returns a dictionary or 
                list structure suitable for final consumption.
        """
        if isinstance(outputs, list):
            formatted_outputs = []
            for output in outputs:
                if hasattr(output, "model_dump"):
                    formatted_outputs.append(output.model_dump())
                else:
                    formatted_outputs.append(output)
            return {'chunked_responses': formatted_outputs}
        elif isinstance(outputs, dict):
            return outputs
        else:
            # single item
            if hasattr(outputs, "model_dump"):
                return outputs.model_dump()
            else:
                return {'chunked_responses': []}

    @abstractmethod
    def _transform_image(self, system_instruction: str, img_path: str, **kwargs) -> Any:
        """
        Hook for child classes to define how to convert an image path 
        into model-ready content. 
        GPT => base64 dict, 
        Gemini => Part.from_image, etc.

        Args:
            img_path (str): The file path to an image on disk.
            system_instruction (str): Optional system or context text to prefix.

        Returns:
            Any:
                Typically a list with [system_text, image_object].
        """
        raise NotImplementedError("Child must define how to transform images to content.")

    # -------------------------------------------------------------------------
    # ============== VISION/TEXT/MERGED INPUTS HANDLER ==============
    # -------------------------------------------------------------------------
    def _handle_vision_inputs(
        self,
        system_instruction: str,
        image_list: list,
        response_format: Any,
        merge_response_format: Any,
        response_format_flag: bool,
        merge_system_prompt: str,
        max_retries: int,
        metadata_prompt_fields: List[str],
        jitter: bool,
        **kwargs
    ) -> Any:
        """
        Handles scenario 2: a list of image file paths for vision-based inference.
        For each image:
        1. Verifies that the file exists.
        2. Attempts to transform the image via _transform_image().
            If an exception is raised, logs the error and skips that image.
        3. Calls _retry_inference() on the transformed image content.
        4. Collects partial responses.
        5. If multiple partial responses exist and a merge_system_prompt is provided,
            combines them using _combine_partial_responses(); otherwise, returns the single result.

        Args:
            system_instruction (str): The context or system instruction to prepend.
            image_list (list): A list of dictionaries containing image file paths and metadata
            response_format (Any): Schema or pydantic model for structured output, or None.
            merge_response_format (Any): Schema or pydantic model for merge prompt structured output, or None.
            response_format_flag (bool): True to attempt structured output; otherwise raw text.
            merge_system_prompt (str): Instructions for merging partial responses.
            max_retries (int): Maximum number of attempts for each inference call.
            jitter (bool): Whether to add random jitter to exponential backoff.
            metadata_prompt_fields (List[str]): List of metadata fields to include in the final response.
            **kwargs: Additional arguments for _retry_inference or _transform_image.

        Returns:
            Union[Dict[str, Any], str, List[Any]]:
                - If multiple valid partial responses and a merge prompt are provided: a merged final result.
                - If only one valid image: that single response.
                - Otherwise, a dictionary with key 'chunked_responses' containing the list of partial responses.
        """
        partials: List[Any] = []
        for i, img_path in enumerate(image_list):
            self.logger.info(f"Processing image {i+1}/{len(image_list)} => {img_path}")

            # Check if the file exists
            if not os.path.exists(img_path):
                self.logger.error(f"Image file not found: {img_path}. Skipping this image.")
                continue

            try:
                single_content = self._transform_image(
                    system_instruction=system_instruction,
                    img_path=img_path,
                    **kwargs
                )
            except Exception as e:
                self.logger.error(f"Error transforming image {img_path}: {e}. Skipping this image.")
                continue

            try:
                resp = self._retry_inference(
                    content_parts=single_content,
                    response_format=response_format,
                    response_format_flag=response_format_flag,
                    max_retries=max_retries,
                    jitter=jitter,
                    **kwargs
                )
                
            except Exception as e:
                self.logger.error(f"Inference error for image {img_path}: {e}. Skipping this image.")
                continue

            try:
                # Get file suffix for image metadata
                file_suffix = img_path.split(".")[-1]
                
                # Open JSON metadata for image
                with open(img_path.replace(file_suffix,"json")) as file:
                    img_metadata = json.load(file)

                # Load desired image metadata into produced response
                for key in metadata_prompt_fields:
                    resp[key] = img_metadata.get(key,"")
                
            except Exception as e:
                self.logger.error(f"Metadata structured output injection error for image {img_path}: {e}. Skipping this image.")
                continue

            partials.append(resp)

        if merge_system_prompt and len(partials) > 1:
            return self._combine_partial_responses(
                partials=partials,
                base_system_prompt=system_instruction,
                merge_system_prompt=merge_system_prompt,
                response_format=merge_response_format,
                response_format_flag=response_format_flag,
                max_retries=max_retries,
                jitter=jitter,
                **kwargs
            )
        else:
            return partials

    def _generate_merged_text_response(
        self,
        system_instruction: str,
        user_text: str,
        response_format: Any,
        response_format_flag: bool,
        max_retries: int,
        jitter: bool,
        max_chunk_tokens: int,
        merge_system_prompt: str,
        **kwargs
    ) -> Union[str, Dict[str, Any], None]:
        """
        Handles scenario 2: text-based partial merges with optional chunking.

        If `max_chunk_tokens` > 0, we split `user_text` into chunks (via `chunk_content`), 
        gather partial responses for each chunk, and then:
        - If more than one partial is collected, we combine them with `_combine_partial_responses`
            using `merge_system_prompt`.
        - If exactly one partial (or none), we simply return that partial (or None if empty).

        If `max_chunk_tokens` == 0, we do a single inference call but still pass `merge_system_prompt`.
        Typically merging won't do much if there's only one response, but it's included for consistency.

        Args:
            system_instruction (str):
                A "system" or background instruction for the model context.
            user_text (str):
                The text content to parse and possibly chunk.
            response_format (Any):
                If structured output is desired (e.g., a pydantic model), 
                otherwise None for raw text.
            response_format_flag (bool):
                True => attempt structured parse; False => raw text.
            max_retries (int):
                Maximum attempts for inference if errors occur.
            jitter (bool):
                Whether to apply random jitter to exponential backoff in `_retry_inference`.
            max_chunk_tokens (int):
                If > 0, chunk `user_text` into smaller segments. 
            merge_system_prompt (str):
                A specialized prompt used to merge partial responses (if more than one partial).
            **kwargs:
                Additional arguments (e.g., temperature) for `_retry_inference`.

        Returns:
            Union[str, Dict[str, Any], None]:
                - If multiple chunks, returns a merged dict or raw string (depending on parse).
                - If a single chunk or no chunk, returns a single partial response (dict or str).
                - If no partial was collected, returns None.
        """
        # If chunking is enabled
        if max_chunk_tokens > 0:
            chunks = self.chunk_content(user_text, max_chunk_tokens)
            partials = []
            for chunk in chunks:
                content_parts = [system_instruction, chunk] if system_instruction else [chunk]
                resp = self._retry_inference(
                    content_parts=content_parts,
                    response_format=response_format,
                    response_format_flag=response_format_flag,
                    max_retries=max_retries,
                    jitter=jitter,
                    **kwargs
                )
                partials.append(resp)

            if len(partials) > 1:
                # Combine partial responses with a final call
                return self._combine_partial_responses(
                    partials=partials,
                    base_system_prompt=system_instruction,
                    merge_system_prompt=merge_system_prompt,
                    response_format=response_format,
                    response_format_flag=response_format_flag,
                    max_retries=max_retries,
                    jitter=jitter,
                    **kwargs
                )
            else:
                # Only one partial or empty => no real merge
                return partials[0] if partials else None
        else:
            # No chunking => single pass
            content_parts = [system_instruction, user_text] if system_instruction else [user_text]
            return self._retry_inference(
                content_parts=content_parts,
                response_format=response_format,
                response_format_flag=response_format_flag,
                max_retries=max_retries,
                jitter=jitter,
                **kwargs
            )

    def _generate_single_text_response(
        self,
        system_instruction: str,
        user_text: str,
        response_format: Any,
        response_format_flag: bool,
        max_retries: int,
        jitter: bool,
        max_chunk_tokens: int,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Handles scenario 3: single pass or chunked text calls with no merges.

        - If `max_chunk_tokens` > 0, we chunk `user_text`, gather partial 
        responses for each chunk, and return them as:
            - A single partial if exactly one chunk,
            - or a concatenated string if multiple chunks (for demonstration).
        - If `max_chunk_tokens` == 0, we do a single inference call.

        Note: There's no merging logic here; partial responses from chunking 
        are simply concatenated or returned as one partial if only one chunk.

        Args:
            system_instruction (str):
                A "system" or background instruction for the model context.
            user_text (str):
                The text content to parse, possibly chunked.
            response_format (Any):
                If structured output is desired (e.g., a pydantic model), 
                otherwise None for raw text.
            response_format_flag (bool):
                True => attempt structured parse; False => raw text.
            max_retries (int):
                Maximum attempts for inference if errors occur.
            jitter (bool):
                Whether to apply random jitter to exponential backoff.
            max_chunk_tokens (int):
                If > 0, chunk `user_text` via `chunk_content`; otherwise do a single pass.
            **kwargs:
                Additional arguments (e.g. temperature) for `_retry_inference`.

        Returns:
            Union[str, Dict[str, Any]]:
                - If only one chunk or no chunking, returns a single partial (str or dict).
                - If multiple chunks, returns a concatenated string of partial responses.
        """
        if max_chunk_tokens > 0:
            chunks = self.chunk_content(user_text, max_chunk_tokens)
            partials = []
            for chunk in chunks:
                content_parts = [system_instruction, chunk] if system_instruction else [chunk]
                resp = self._retry_inference(
                    content_parts=content_parts,
                    response_format=response_format,
                    response_format_flag=response_format_flag,
                    max_retries=max_retries,
                    jitter=jitter,
                    **kwargs
                )
                partials.append(resp)

            if len(partials) == 1:
                # Only one chunk => single partial
                return partials[0]
            else:
                # multiple partial outputs => no merge => return them as single concatenated string
                return "\n".join(str(partial) for partial in partials)
        else:
            # Single pass, no merges
            content_parts = [system_instruction, user_text] if system_instruction else [user_text]
            return self._retry_inference(
                content_parts=content_parts,
                response_format=response_format,
                response_format_flag=response_format_flag,
                max_retries=max_retries,
                jitter=jitter,
                **kwargs
            )

    def generate_response(
        self,
        system_instruction: str,
        user_instruction: Union[str, List[str]],
        response_format: Any = None,
        merge_response_format: Any = None,
        response_format_flag: bool = False,
        max_retries: int = 3,
        jitter: bool = True,
        merge_system_prompt: str = "",
        use_llm_vision: bool = False,
        max_chunk_tokens: int = 0,
        metadata_prompt_fields: List[str] = [],
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Master orchestration method for generating a response from the LLM. 
        Dispatches to three main scenarios:

        1) Vision scenario (`use_llm_vision=True`):
        - user_instruction must be a list of image paths
        - calls `_handle_vision_inputs` -> possibly merges partials
        - returns final output via `_format_outputs`

        2) Merged text scenario (not vision, but `merge_system_prompt != ""`):
        - calls `_generate_merged_text_response` which can chunk text if needed,
            gather partials, and unify them with a final merge call.

        3) Single pass scenario (everything else):
        - calls `_generate_single_text_response` for either a single pass or chunked 
            text calls that are returned as-is without merges.

        Args:
            system_instruction (str):
                A "system" or background instruction for the model context.
            user_instruction (Union[str, List[str]]):
                - If a string: text content
                - If a list of strings: image paths for vision scenario
            response_format (Any):
                A pydantic model or other schema if structured output is desired, 
                otherwise None for raw text.
            merge_response_format (Any):
                A pydantic model or other schema if structured output is desired for merge prompt, 
                otherwise None for raw text.
            response_format_flag (bool):
                True => attempt structured parse, else raw text.
            max_retries (int):
                Max attempts in case of transient inference errors.
            jitter (bool):
                Whether to apply random jitter in exponential backoff for `_retry_inference`.
            merge_system_prompt (str):
                If non-empty, triggers partial merges in text scenario. 
            use_llm_vision (bool):
                If True, interpret user_instruction as images. 
            max_chunk_tokens (int):
                If >0, chunk user text by tokens for partial inference calls.
            metadata_prompt_fields (List[str]):
                List of metadata fields to include in the final response.
            **kwargs:
                Additional arguments (e.g., temperature) for `_retry_inference` or child logic.

        Returns:
            Union[str, Dict[str, Any]]:
                - Vision scenario typically ends up returning a dict if merges or partials,
                or raw text/string if structured parse is false,
                all passed through `_format_outputs`.
                - Merged text scenario => a dict if partial merges are done,
                or possibly a single partial (str/dict).
                - Single pass => can return a single str/dict or a concatenated string 
                if multiple chunked partials are used.
        """
        if use_llm_vision:
            # SCENARIO 1: Vision
            if not isinstance(user_instruction, list):
                raise ValueError("For vision mode, user_instruction must be a list of dictionaries containing image paths and metadata.")
            
            partial_or_merged = self._handle_vision_inputs(
                system_instruction=system_instruction,
                image_list=user_instruction,
                response_format=response_format,
                merge_response_format=merge_response_format,
                response_format_flag=response_format_flag,
                merge_system_prompt=merge_system_prompt,
                max_retries=max_retries,
                jitter=jitter,
                metadata_prompt_fields=metadata_prompt_fields,
                **kwargs
            )
            return self._format_outputs(partial_or_merged)

        elif isinstance(user_instruction, str) and merge_system_prompt:
            # SCENARIO 2: Merged text
            merged_resp = self._generate_merged_text_response(
                system_instruction=system_instruction,
                user_text=user_instruction,
                response_format=response_format,
                response_format_flag=response_format_flag,
                max_retries=max_retries,
                jitter=jitter,
                max_chunk_tokens=max_chunk_tokens,
                merge_system_prompt=merge_system_prompt,
                **kwargs
            )
            return self._format_outputs(merged_resp)

        else:
            # SCENARIO 3: Single pass
            if not isinstance(user_instruction, str):
                raise ValueError("For text mode, user_instruction must be a string.")

            single_resp = self._generate_single_text_response(
                system_instruction=system_instruction,
                user_text=user_instruction,
                response_format=response_format,
                response_format_flag=response_format_flag,
                max_retries=max_retries,
                jitter=jitter,
                max_chunk_tokens=max_chunk_tokens,
                **kwargs
            )
            return single_resp
