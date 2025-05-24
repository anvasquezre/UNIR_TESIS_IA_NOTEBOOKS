from typing import Callable, Optional

import tenacity
from langchain.llms import BaseLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from core.logger import get_logger
from src.extraction.extraction_responses import LLMResponse
from src.extraction.extraction_template import Base64Image, create_multimodal_template
from src.extraction.output_parser import LLMOutputParser

logger = get_logger(__name__)


class LLMExtractor:
    def __init__(
        self,
        llm: BaseLLM,
        output_parser: LLMOutputParser,
        extraction_template: ChatPromptTemplate,
    ):
        self.llm = llm
        self.output_parser = output_parser
        self.extraction_template = extraction_template

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    def extract(self, context: str, question: str) -> LLMResponse:
        prompt_value = self._get_prompt_value(context, question)
        answer: AIMessage = self.llm.invoke(input=prompt_value)
        return self.output_parser.parse(answer.content)

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    async def aextract(self, context: str, question: str) -> LLMResponse:
        prompt_value = self._get_prompt_value(context, question)
        answer: AIMessage = await self.llm.ainvoke(input=prompt_value)
        return await self.output_parser.aparse(answer.content)

    def _get_prompt_value(
        self,
        context: str,
        question: str,
    ) -> str:
        return self.extraction_template.invoke(
            {
                "context": context,
                "question": question,
                "format": self.output_parser.pydantic_output_parser.get_format_instructions(),
            }
        )


class LLMExtractorMultimodal(LLMExtractor):
    def __init__(
        self,
        llm: BaseLLM,
        output_parser: LLMOutputParser,
        image_template_func: Callable = create_multimodal_template,
        extraction_template: ChatPromptTemplate = None,
    ):
        super().__init__(llm, output_parser, extraction_template)
        self.image_template_func = image_template_func

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    def extract(
        self,
        question: str,
        context: Optional[str] = None,
        image: Optional[Base64Image] = None,
    ) -> LLMResponse:
        if not context and not image:
            raise ValueError("Either context or image must be provided.")
        prompt_value = self._get_prompt_value(
            question=question, context=context, image=image
        )
        logger.warning(f"Prompt value: {prompt_value}")
        answer: AIMessage = self.llm.invoke(input=prompt_value)
        return self.output_parser.parse(answer.content)

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    async def aextract(
        self,
        question: str,
        context: Optional[str] = None,
        image: Optional[Base64Image] = None,
    ) -> LLMResponse:
        if not context and not image:
            raise ValueError("Either context or image must be provided.")
        prompt_value = self._get_prompt_value(
            question=question, context=context, image=image
        )
        logger.warning(f"Prompt value: {prompt_value}")
        answer: AIMessage = await self.llm.ainvoke(input=prompt_value)
        return await self.output_parser.aparse(answer.content)

    def _get_prompt_value(
        self,
        question: str,
        context: Optional[str] = None,
        image: Optional[Base64Image] = None,
    ) -> str:

        extraction_template = (
            self.image_template_func(image) if image else self.extraction_template
        )

        template_kwargs = {
            "question": question,
            "format": self.output_parser.pydantic_output_parser.get_format_instructions(),
        }

        if context:
            template_kwargs["context"] = context
        return extraction_template.invoke(input=template_kwargs)
