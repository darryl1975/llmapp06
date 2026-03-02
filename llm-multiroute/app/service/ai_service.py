import json
import re
import time
from typing import Optional

import httpx

from app.config import settings
from app.dto.classification_response import ClassificationResponse
from app.dto.intent_response import IntentResponse
from app.dto.sentiment_response import SentimentResponse
from app.dto.summary_response import SummaryResponse
from app.monitoring import metrics_store
from app.router.model_router import ModelRouter, TaskType, model_router


class AIService:
    def __init__(
        self,
        http_client: Optional[httpx.Client] = None,
        router: Optional[ModelRouter] = None,
    ):
        self.http_client = http_client or httpx.Client(timeout=120.0)
        self.base_url = settings.OLLAMA_BASE_URL
        self.temperature = settings.OLLAMA_TEMPERATURE
        self.api_key = settings.OLLAMA_API_KEY
        self.router = router or model_router

    def _chat(self, prompt: str, model: str, task_type: str = "unknown") -> str:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start = time.monotonic()
        response = self.http_client.post(
            f"{self.base_url}/api/chat",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": self.temperature,
            },
        )
        response.raise_for_status()
        latency = time.monotonic() - start

        body = response.json()

        # Record latency
        metrics_store.record_latency(task_type, model, latency)

        # Record token usage (Ollama returns prompt_eval_count / eval_count)
        input_tokens = body.get("prompt_eval_count", 0)
        output_tokens = body.get("eval_count", 0)
        metrics_store.record_token_usage(
            task_type, model, input_tokens, output_tokens, input_tokens + output_tokens
        )

        return body["message"]["content"]

    def classify_text(self, text: str) -> ClassificationResponse:
        model = self.router.get_model(TaskType.CLASSIFY)
        prompt = (
            "Analyze the following text and classify it with appropriate labels and tags. "
            "Respond with ONLY valid JSON, no additional text or explanation.\n\n"
            f"Text: {text}\n\n"
            "Return JSON in this exact format:\n"
            '{"labels": ["label1", "label2"], "primaryCategory": "category", "confidence": 0.9}'
        )
        response = self._chat(prompt, model, task_type="classify")
        return self._parse_json(response, ClassificationResponse)

    def analyze_sentiment(self, text: str) -> SentimentResponse:
        model = self.router.get_model(TaskType.SENTIMENT)
        prompt = (
            "Analyze the sentiment of the following text. "
            "Respond with ONLY valid JSON, no additional text or explanation.\n\n"
            f"Text: {text}\n\n"
            "Return JSON in this exact format:\n"
            '{"overallSentiment": "positive", "sentimentScore": 0.8, '
            '"emotions": ["joy", "excitement"], "confidence": 0.9}'
        )
        response = self._chat(prompt, model, task_type="sentiment")
        return self._parse_json(response, SentimentResponse)

    def summarize_text(self, text: str) -> SummaryResponse:
        model = self.router.get_model(TaskType.SUMMARIZE)
        prompt = (
            "Summarize the following text concisely. "
            "Respond with ONLY valid JSON, no additional text or explanation.\n\n"
            f"Text: {text}\n\n"
            "Return JSON in this exact format:\n"
            '{"summary": "your summary here", "keyPoints": ["point1", "point2", "point3"], "wordCount": 25}'
        )
        response = self._chat(prompt, model, task_type="summarize")
        return self._parse_json(response, SummaryResponse)

    def detect_intent(self, text: str) -> IntentResponse:
        model = self.router.get_model(TaskType.INTENT)
        prompt = (
            "Detect the intent behind the following text. "
            "Respond with ONLY valid JSON, no additional text or explanation.\n\n"
            f"Text: {text}\n\n"
            "intentCategory must be exactly one of:\n"
            "- question: the text asks for information\n"
            "- request: the text politely asks someone to do something\n"
            "- command: the text gives a direct order or instruction\n"
            "- statement: the text declares facts or information\n\n"
            "Return JSON in this exact format:\n"
            '{"primaryIntent": "main_intent", "secondaryIntents": ["intent1", "intent2"], '
            '"intentCategory": "question", "confidence": 0.9}'
        )
        response = self._chat(prompt, model, task_type="intent")
        return self._parse_json(response, IntentResponse)

    @staticmethod
    def _parse_json(raw: str, model_class: type):
        cleaned = raw.strip()
        # Strip markdown code blocks if present
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
        try:
            data = json.loads(cleaned)
            return model_class(**data)
        except Exception as e:
            raise RuntimeError(f"Failed to parse AI response as JSON: {raw}") from e
