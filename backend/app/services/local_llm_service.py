from dataclasses import dataclass

import httpx

from backend.app.core.config import Settings
from backend.app.schemas.ai_insight_schema import LLMStatusResponse


@dataclass
class LLMTextResult:
    content: str
    source: str
    online: bool


class LocalLLMService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def check_status(self) -> LLMStatusResponse:
        # LLM 預設可關閉，避免開發者沒啟動模型時整個系統失敗。
        if not self.settings.llm_enabled:
            return LLMStatusResponse(
                enabled=False,
                online=False,
                base_url=self.settings.llm_base_url,
                model=self.settings.llm_model,
                message="LLM integration is disabled. Set LLM_ENABLED=true to enable it.",
            )

        try:
            response = httpx.get(
                f"{self.settings.llm_base_url}/v1/models",
                timeout=3,
            )
            response.raise_for_status()
        except Exception as exc:
            return LLMStatusResponse(
                enabled=True,
                online=False,
                base_url=self.settings.llm_base_url,
                model=self.settings.llm_model,
                message=f"LLM server is unreachable: {exc}",
            )

        return LLMStatusResponse(
            enabled=True,
            online=True,
            base_url=self.settings.llm_base_url,
            model=self.settings.llm_model,
            message="LLM server is online.",
        )

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback_content: str,
    ) -> LLMTextResult:
        # 如果 LLM 沒啟用，直接回傳 fallback，讓 API response 穩定。
        if not self.settings.llm_enabled:
            return LLMTextResult(
                content=fallback_content,
                source="fallback",
                online=False,
            )

        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }

        try:
            response = httpx.post(
                f"{self.settings.llm_base_url}/v1/chat/completions",
                json=payload,
                timeout=self.settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except Exception:
            return LLMTextResult(
                content=fallback_content,
                source="fallback",
                online=False,
            )

        return LLMTextResult(
            content=content.strip(),
            source="llm",
            online=True,
        )
