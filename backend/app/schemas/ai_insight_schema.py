from pydantic import BaseModel, Field


class LLMStatusResponse(BaseModel):
    enabled: bool = Field(description="Whether LLM integration is enabled")
    online: bool = Field(description="Whether LLM server is reachable")
    base_url: str = Field(description="LLM server base URL")
    model: str = Field(description="Configured LLM model name")
    message: str = Field(description="Status message")


class AIInsightRequest(BaseModel):
    user_goal: str | None = Field(
        default=None,
        description="Optional user goal for generating a more focused explanation",
    )


class AIInsightResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    title: str = Field(description="Insight title")
    content: str = Field(description="Generated or fallback explanation")
    source: str = Field(description="llm or fallback")
    model: str = Field(description="LLM model name or fallback")
    llm_enabled: bool = Field(description="Whether LLM integration is enabled")
    llm_online: bool = Field(description="Whether LLM server is online")
