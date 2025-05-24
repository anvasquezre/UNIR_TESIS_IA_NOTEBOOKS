from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    answer: str = Field(default="", description="The answer to the question.")
