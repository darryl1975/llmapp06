from fastapi import APIRouter

from app.dto.classification_response import ClassificationResponse
from app.dto.intent_response import IntentResponse
from app.dto.sentiment_response import SentimentResponse
from app.dto.summary_response import SummaryResponse
from app.dto.text_request import TextRequest
from app.monitoring import metrics_store, safety_checker
from app.router.model_router import model_router
from app.service.ai_service import AIService

router = APIRouter(prefix="/api/ai", tags=["AI Text Analysis"])

ai_service = AIService()


def _check_safety(text: str, task_type: str) -> None:
    """Run prompt injection check and record any detections."""
    result = safety_checker.check_prompt_injection(text)
    if result["detected"]:
        metrics_store.record_safety_event(
            event_type="prompt_injection",
            task_type=task_type,
            input_text_snippet=text,
            details=f"Matched patterns: {result['patterns']}",
        )


def _check_response_policy(text: str, response_text: str, task_type: str) -> None:
    """Run policy violation check on input+output and record any detections."""
    result = safety_checker.check_policy_violations(text, response_text)
    if result["detected"]:
        metrics_store.record_safety_event(
            event_type="policy_violation",
            task_type=task_type,
            input_text_snippet=text,
            details=f"Input matches: {result['input_matches']}, Output matches: {result['output_matches']}",
        )


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify Text",
    description="Analyzes text and returns classification labels, tags, and primary category",
)
def classify_text(request: TextRequest) -> ClassificationResponse:
    _check_safety(request.text, "classify")
    result = ai_service.classify_text(request.text)
    _check_response_policy(request.text, str(result), "classify")
    return result


@router.post(
    "/sentiment",
    response_model=SentimentResponse,
    summary="Analyze Sentiment",
    description="Analyzes text sentiment (positive, negative, neutral) and detects specific emotions",
)
def analyze_sentiment(request: TextRequest) -> SentimentResponse:
    _check_safety(request.text, "sentiment")
    result = ai_service.analyze_sentiment(request.text)
    _check_response_policy(request.text, str(result), "sentiment")
    return result


@router.post(
    "/summarize",
    response_model=SummaryResponse,
    summary="Summarize Text",
    description="Generates a concise summary with key points from the provided text",
)
def summarize_text(request: TextRequest) -> SummaryResponse:
    _check_safety(request.text, "summarize")
    result = ai_service.summarize_text(request.text)
    _check_response_policy(request.text, str(result), "summarize")
    return result


@router.post(
    "/intent",
    response_model=IntentResponse,
    summary="Detect Intent",
    description="Identifies the intent and purpose behind the text (question, request, statement, command)",
)
def detect_intent(request: TextRequest) -> IntentResponse:
    _check_safety(request.text, "intent")
    result = ai_service.detect_intent(request.text)
    _check_response_policy(request.text, str(result), "intent")
    return result


@router.get(
    "/routes",
    summary="Get Route Configuration",
    description="Returns the current model routing table showing which model handles each task type",
)
def get_routes() -> dict[str, str]:
    return model_router.get_routes()


# ── Metrics endpoints ─────────────────────────────────────────────


@router.get(
    "/metrics/cost",
    summary="Get Cost Metrics",
    description="Returns token usage records and summary totals",
)
def get_cost_metrics() -> dict:
    return metrics_store.get_cost_metrics()


@router.get(
    "/metrics/performance",
    summary="Get Performance Metrics",
    description="Returns latency records with p50/p95/avg statistics",
)
def get_performance_metrics() -> dict:
    return metrics_store.get_performance_metrics()


@router.get(
    "/metrics/safety",
    summary="Get Safety Metrics",
    description="Returns prompt injection and policy violation records",
)
def get_safety_metrics() -> dict:
    return metrics_store.get_safety_metrics()
