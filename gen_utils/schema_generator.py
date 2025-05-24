# schema_generator.py

import logging
import StructuredOutput.DynamicPydanticClasses as dynamic_models
from typing import Optional, Union
from models.gpt_model import GPTModel
from models.gemini_model import GeminiModel

logger = logging.getLogger(__name__)

class SchemaGenerator:
    """
    Dynamically generates Pydantic classes based on LLM output.
    This can work with either GPTModel or GeminiModel, as long as the LLM 
    provides a generate_response() method returning code as raw text.
    """

    def __init__(
        self,
        system_instruction: str,
        llm: Optional[Union[GPTModel, GeminiModel]] = None, 
        max_retries: int = 3
    ) -> None:
        """
        Args:
            system_instruction (str): A prompt telling the LLM how to produce code (the schema).
            llm: The language model to use (GPTModel or GeminiModel).
            max_retries: Number of attempts to get valid code from the LLM.
        """
        self._system_instruction = system_instruction
        self._llm = llm or GPTModel()  # Default to GPTModel if none passed
        self._max_retries = max_retries
        logger.debug("SchemaGenerator initialized.")

    def generate_schema(self, user_instruction: str) -> None:
        """
        Calls the LLM (GPT or Gemini) to generate Pydantic class definitions,
        then executes them in the dynamic_models module.

        Args:
            user_instruction (str): Typically describes fields or data we want in the schema.
        """
        logger.info("Generating dynamic schema via LLM...")

        retries = 0
        while retries < self._max_retries:
            try:
                # 1) Get code snippet from LLM
                response_raw = self._llm.generate_response(
                    system_instruction=self._system_instruction,
                    user_instruction=user_instruction,
                    # We want raw text for code, not JSON
                    response_format_flag=False
                )

                # 2) Clean out triple backticks, 'python' fences, etc.
                code_cleaned = (
                    response_raw
                    .replace("```", "")
                    .replace("python", "")
                    .strip()
                )
                logger.debug(f"Generated schema code:\n{code_cleaned}")

                # 3) Execute code in dynamic_models
                exec(code_cleaned, dynamic_models.__dict__)

                # Optionally, write to file for reference
                with open('./StructuredOutput/DynamicPydanticClasses.py', 'w', encoding='utf-8') as file:
                    file.write(code_cleaned)

                logger.info("Schema code executed successfully; classes are now in dynamic_models.")
                return

            except Exception as e:
                logger.error(f"Error executing schema code: {e}")
                retries += 1
                if retries >= self._max_retries:
                    raise RuntimeError(
                        f"Schema code execution failed after {self._max_retries} retries. Last error: {e}"
                    )
