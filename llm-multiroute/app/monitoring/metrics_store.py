import json
import os
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path


METRICS_DIR = Path(__file__).resolve().parent.parent.parent / "metrics"

COST_FILE = METRICS_DIR / "cost_metrics.json"
PERFORMANCE_FILE = METRICS_DIR / "performance_metrics.json"
SAFETY_FILE = METRICS_DIR / "safety_metrics.json"


class MetricsStore:
    def __init__(self):
        self._lock = threading.Lock()
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── JSON I/O helpers ──────────────────────────────────────────

    @staticmethod
    def _load(filepath: Path) -> dict:
        if filepath.exists():
            with open(filepath, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def _save(filepath: Path, data: dict) -> None:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    # ── Cost / token usage ────────────────────────────────────────

    def record_token_usage(
        self,
        task_type: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ) -> None:
        with self._lock:
            data = self._load(COST_FILE)
            if "records" not in data:
                data["records"] = []
                data["summary"] = {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_requests": 0,
                }

            data["records"].append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "task_type": task_type,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            )

            data["summary"]["total_input_tokens"] += input_tokens
            data["summary"]["total_output_tokens"] += output_tokens
            data["summary"]["total_requests"] += 1

            self._save(COST_FILE, data)

    # ── Performance / latency ─────────────────────────────────────

    def record_latency(
        self, task_type: str, model: str, latency_seconds: float
    ) -> None:
        with self._lock:
            data = self._load(PERFORMANCE_FILE)
            if "records" not in data:
                data["records"] = []
                data["summary"] = {
                    "p50_latency": 0.0,
                    "p95_latency": 0.0,
                    "avg_latency": 0.0,
                    "total_requests": 0,
                }

            data["records"].append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "task_type": task_type,
                    "model": model,
                    "latency_seconds": round(latency_seconds, 4),
                }
            )

            latencies = [r["latency_seconds"] for r in data["records"]]
            data["summary"]["p50_latency"] = round(statistics.median(latencies), 4)
            data["summary"]["p95_latency"] = round(
                statistics.quantiles(latencies, n=20)[-1]
                if len(latencies) >= 2
                else latencies[0],
                4,
            )
            data["summary"]["avg_latency"] = round(statistics.mean(latencies), 4)
            data["summary"]["total_requests"] = len(latencies)

            self._save(PERFORMANCE_FILE, data)

    # ── Safety events ─────────────────────────────────────────────

    def record_safety_event(
        self,
        event_type: str,
        task_type: str,
        input_text_snippet: str,
        details: str,
    ) -> None:
        with self._lock:
            data = self._load(SAFETY_FILE)
            if "records" not in data:
                data["records"] = []
                data["summary"] = {
                    "total_prompt_injection_attempts": 0,
                    "total_policy_violations": 0,
                }

            data["records"].append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": event_type,
                    "task_type": task_type,
                    "input_text_snippet": input_text_snippet[:200],
                    "details": details,
                }
            )

            if event_type == "prompt_injection":
                data["summary"]["total_prompt_injection_attempts"] += 1
            elif event_type == "policy_violation":
                data["summary"]["total_policy_violations"] += 1

            self._save(SAFETY_FILE, data)

    # ── Read-only access for API endpoints ────────────────────────

    def get_cost_metrics(self) -> dict:
        with self._lock:
            return self._load(COST_FILE) or {
                "records": [],
                "summary": {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_requests": 0,
                },
            }

    def get_performance_metrics(self) -> dict:
        with self._lock:
            return self._load(PERFORMANCE_FILE) or {
                "records": [],
                "summary": {
                    "p50_latency": 0.0,
                    "p95_latency": 0.0,
                    "avg_latency": 0.0,
                    "total_requests": 0,
                },
            }

    def get_safety_metrics(self) -> dict:
        with self._lock:
            return self._load(SAFETY_FILE) or {
                "records": [],
                "summary": {
                    "total_prompt_injection_attempts": 0,
                    "total_policy_violations": 0,
                },
            }
