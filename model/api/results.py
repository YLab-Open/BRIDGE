from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QueryResult:
    """Normalized single-sample API result."""

    custom_id: str
    text: str = ""
    raw: dict[str, Any] | None = None
    error: str = ""
    summary: str = ""
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_trace_dict(self, include_raw: bool = True) -> dict[str, Any]:
        trace = {
            "custom_id": self.custom_id,
            "output": self.text,
            "summary": self.summary,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }
        if include_raw:
            trace["raw"] = self.raw or {}
        return trace


@dataclass
class ExperimentResult:
    """Summary for one provider/model/task/prompt experiment."""

    result_path: Path
    trace_dir: Path
    provider: str
    model_name: str
    task_name: str
    prompt_mode: str
    decoding_strategy: str
    seed: int
    num_samples: int
    num_errors: int = 0
    error_examples: list[str] | None = None
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        return self.num_errors == 0


@dataclass
class ApiRunResult:
    """Summary for a full API run across all configured experiments."""

    provider: str
    model_names: list[str]
    experiments: list[ExperimentResult]

    @property
    def num_errors(self) -> int:
        return sum(item.num_errors for item in self.experiments)

    @property
    def result_paths(self) -> list[Path]:
        return [item.result_path for item in self.experiments]


def summarize_query_results(results: list[QueryResult], limit: int = 5) -> tuple[int, list[str]]:
    """Return total error count and a few compact error examples."""
    examples = []
    for result in results:
        if result.error:
            examples.append(f"{result.custom_id}: {result.error}")
    return len(examples), examples[:limit]
