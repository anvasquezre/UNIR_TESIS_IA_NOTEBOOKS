import re
from typing import Optional, Type

from json_repair import repair_json
from langchain.llms import BaseLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from core.constants import OUTPUT_PARSER_PROMPT
from core.logger import get_logger

logger = get_logger(__name__)

_json_markdown_re = re.compile(r"```(json)?(.*)", re.DOTALL)


PROMPT = OUTPUT_PARSER_PROMPT


class LLMOutputParser:
    prompt: str = PROMPT

    def __init__(
        self,
        serializable: Type[BaseModel],
        llm: BaseLLM,
        max_retries: int = 3,
    ):
        self.serializable = serializable
        self.pydantic_output_parser = PydanticOutputParser(pydantic_object=serializable)
        self.llm = llm
        if max_retries < 1:
            raise ValueError("max_retries must be a positive integer")
        self.max_retries = max_retries

    async def aparse(self, llm_output: str) -> BaseModel:
        return await self._aparse(llm_output, 0)

    def parse(self, llm_output: str) -> BaseModel:
        return self._parse(llm_output, 0)

    async def _aparse(self, llm_output: str, attemp: int) -> BaseModel:
        if attemp >= self.max_retries:
            logger.warning("Max retries reached while trying to parse agent reasoning")
            return self.serializable(unparsed_output=llm_output)
        try:
            agent_reasoning = self._basic_parse(llm_output)
        except Exception as e:
            logger.warning("Error parsing agent reasoning, forcing output to be fixed")
            logger.warning(f"Agent reasoning: {llm_output}")
            agent_reasoning = await self._ahandle_error_with_recursion(
                llm_output, attemp + 1, error=e
            )
        return agent_reasoning

    def _basic_parse(self, llm_output: str) -> BaseModel:
        json_string = get_json_markdown_code_block(llm_output)
        fixed_json = repair_json(json_string, skip_json_loads=True)
        if not fixed_json and not json_string:
            # empty string or None
            logger.debug(
                "Fixed JSON is empty or None, returning serializable with unparsed output"
            )
            return self.serializable(unparsed_output=llm_output)
        agent_reasoning = self.pydantic_output_parser.parse(llm_output)
        return agent_reasoning

    async def _ahandle_error_with_recursion(
        self,
        llm_output: str,
        attemp: int = 0,
        error: Optional[Exception] = None,
    ) -> BaseModel:
        PROMPT = self.prompt.format(json=llm_output, e=str(error))
        response: AIMessage = await self.llm.ainvoke(
            [
                SystemMessage(
                    content="You are an expert in JSON formatting. you will fix the JSON formatting error in the agent reasoning. Return only the json as a code block with ```json```."
                ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "text",
                            "text": self.pydantic_output_parser.get_format_instructions(),
                        },
                    ]
                ),
            ]
        )
        agent_reasoning = await self._aparse(response.content, attemp + 1)
        return agent_reasoning

    def _parse(
        self,
        llm_output: str,
        attemp: int = 0,
    ) -> BaseModel:
        if attemp >= self.max_retries:
            logger.warning("Max retries reached while trying to parse agent reasoning")
            return self.serializable(unparsed_output=llm_output)

        try:
            agent_reasoning = self._basic_parse(llm_output)
        except Exception as e:
            logger.warning("Error parsing agent reasoning, forcing output to be fixed")

            agent_reasoning = self.handle_error_with_recursion(
                llm_output, attemp + 1, error=e
            )
        return agent_reasoning

    def handle_error_with_recursion(
        self,
        llm_output: str,
        attemp: int = 0,
        error: Optional[Exception] = None,
    ) -> BaseModel:

        PROMPT = self.prompt.format(json=llm_output, e=str(error))
        response: AIMessage = self.llm.invoke(
            [
                SystemMessage(
                    content="You are an expert in JSON formatting. you will fix the JSON formatting error in the agent reasoning. Return only the json as a code block with ```json```."
                ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "text",
                            "text": self.pydantic_output_parser.get_format_instructions(),
                        },
                    ]
                ),
            ]
        )
        agent_reasoning = self._parse(response.content, attemp + 1)
        return agent_reasoning


def get_json_markdown_code_block(json_string: str) -> str:
    match = _json_markdown_re.search(json_string)
    # If no match found, assume the entire string is a JSON string
    if match is None:
        json_str = json_string
        logger.debug("No match found")
    else:
        # If match found, use the content within the backticks
        json_str = match.group(2)
        logger.debug("Match found")
    return json_str
