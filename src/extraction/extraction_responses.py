from pydantic import BaseModel, ConfigDict, Field


class BaseExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class LLMResponse(BaseExtractionResponse):
    answer: str = Field("", description="The answer to the question.")
