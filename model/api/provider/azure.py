from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from model.api.azure_common import (
    apply_output_token_key,
    build_chat_completion_batch_body,
    build_response_batch_body,
    default_chat_config,
    default_response_config,
    extract_summary_from_responses_api_output,
    extract_text_from_chat_completion_body,
    extract_text_from_responses_api_output,
    is_reasoning_model,
)
from model.api.common import deep_merge_dict, safe_model_dump
from model.api.prompts import custom_id
from model.api.results import QueryResult
from util.io import write_json, write_jsonl

TERMINAL_BATCH_STATUSES = {"completed", "failed", "cancelled", "expired"}


class AzureProvider:
    """Azure OpenAI provider with single and batch execution."""

    provider = "azure"

    def __init__(
        self,
        model_name: str,
        model_cfg: dict[str, Any],
        run_cfg: dict[str, Any],
        api_settings: dict[str, Any],
    ):
        self.model_name = model_name
        self.model_cfg = model_cfg
        self.run_cfg = run_cfg
        self.api_settings = api_settings
        self.deploy_name = model_cfg.get("deploy_name") or model_cfg.get("deployment") or model_name
        self.batch_deploy_name = (
            model_cfg.get("batch_deploy_name")
            or model_cfg.get("batch_deployment")
            or self.deploy_name
        )
        self.support_reasoning = bool(
            model_cfg.get("support_reasoning", is_reasoning_model(self.deploy_name))
        )
        self.support_batch = bool(model_cfg.get("support_batch", False))

    def should_use_batch(self) -> bool:
        requested = str(self.api_settings.get("use_batch", "auto")).lower()
        if requested in {"0", "false", "no", "single"}:
            return False
        if requested in {"1", "true", "yes", "batch"}:
            if not self.support_batch:
                raise ValueError(f"Model {self.model_name} is configured with support_batch=false.")
            return True
        return self.support_batch

    def _client(self):
        from openai import AzureOpenAI

        if not self.api_settings.get("endpoint"):
            raise ValueError("Azure endpoint is missing. Set configs/API.key.yaml or AZURE_OPENAI_ENDPOINT.")
        if not self.api_settings.get("api_key"):
            raise ValueError("Azure API key is missing. Set configs/API.key.yaml or AZURE_OPENAI_API_KEY.")
        timeout = float(self.api_settings.get("timeout", 900) or 900)
        max_retries = int(self.api_settings.get("sdk_max_retries", 2) or 0)
        return AzureOpenAI(
            api_version=str(self.api_settings.get("api_version")),
            azure_endpoint=str(self.api_settings.get("endpoint")),
            api_key=str(self.api_settings.get("api_key")),
            timeout=timeout,
            max_retries=max_retries,
        )

    def query_single(
        self,
        messages: list[dict[str, str]],
        cid: str,
        max_token_output: int,
    ) -> QueryResult:
        """Run one Azure request."""
        t0 = time.time()
        mode = str(self.model_cfg.get("mode") or "").lower()
        if not mode:
            mode = "response" if self.support_reasoning else "chat"
        try:
            client = self._client()
            if mode in {"response", "responses"}:
                config = default_response_config(self.deploy_name, max_token_output)
                config["reasoning"]["summary"] = "detailed"
                deep_merge_dict(config, self.run_cfg.get("responses_params") or {})
                config = apply_output_token_key(
                    config=config,
                    model_name=self.deploy_name,
                    mode="response",
                    max_token_output=max_token_output,
                )
                model = config.pop("model", self.deploy_name)
                response = client.responses.create(model=model, input=messages, **config)
                raw = safe_model_dump(response)
                usage = raw.get("usage") or {}
                return QueryResult(
                    custom_id=cid,
                    text=extract_text_from_responses_api_output(raw),
                    summary=extract_summary_from_responses_api_output(raw),
                    raw=raw,
                    latency_ms=int((time.time() - t0) * 1000),
                    tokens_in=int(usage.get("input_tokens") or 0),
                    tokens_out=int(usage.get("output_tokens") or 0),
                )

            config = default_chat_config(self.deploy_name, max_token_output)
            deep_merge_dict(config, self.run_cfg.get("chat_params") or {})
            config = apply_output_token_key(
                config=config,
                model_name=self.deploy_name,
                mode="chat",
                max_token_output=max_token_output,
            )
            model = config.pop("model", self.deploy_name)
            response = client.chat.completions.create(model=model, messages=messages, **config)
            raw = safe_model_dump(response)
            usage = raw.get("usage") or {}
            return QueryResult(
                custom_id=cid,
                text=extract_text_from_chat_completion_body(raw),
                raw=raw,
                latency_ms=int((time.time() - t0) * 1000),
                tokens_in=int(usage.get("prompt_tokens") or 0),
                tokens_out=int(usage.get("completion_tokens") or 0),
            )
        except Exception as exc:
            return QueryResult(
                custom_id=cid,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.time() - t0) * 1000),
            )

    def build_batch_records(
        self,
        task_name: str,
        prompt_mode: str,
        split: str,
        dataset: list[dict[str, Any]],
        messages_by_id: dict[str, list[dict[str, str]]],
        max_token_output: int,
    ) -> tuple[list[dict[str, Any]], str]:
        """Build Azure batch JSONL records and return the batch endpoint."""
        mode = str(self.model_cfg.get("batch_mode") or self.model_cfg.get("mode") or "").lower()
        if not mode:
            mode = "response" if self.support_reasoning else "chat"
        if mode in {"response", "responses"}:
            endpoint = "/v1/responses"
            params = {"reasoning": {"summary": "detailed"}}
            deep_merge_dict(params, self.run_cfg.get("responses_params") or {})
            builder = build_response_batch_body
        else:
            endpoint = "/v1/chat/completions"
            params = dict(self.run_cfg.get("chat_params") or {})
            builder = build_chat_completion_batch_body

        records = []
        for row in dataset:
            sid = str(row["id"])
            records.append(
                {
                    "custom_id": custom_id(task_name, self.model_name, prompt_mode, split, sid),
                    "method": "POST",
                    "url": endpoint,
                    "body": builder(
                        model_name=self.batch_deploy_name,
                        messages=messages_by_id[sid],
                        max_token_output=max_token_output,
                        dict_config=params,
                    ),
                }
            )
        return records, endpoint

    def run_batch(
        self,
        records: list[dict[str, Any]],
        endpoint: str,
        trace_dir: Path,
        keep_artifacts: bool = True,
    ) -> list[QueryResult]:
        """Upload, create, poll, download, and parse an Azure batch job."""
        trace_dir.mkdir(parents=True, exist_ok=True)
        input_path = trace_dir / "azure_batch_input.jsonl"
        output_path = trace_dir / "azure_batch_output.jsonl"
        error_path = trace_dir / "azure_batch_error.jsonl"
        status_path = trace_dir / "azure_batch_status.jsonl"
        meta_path = trace_dir / "azure_batch_metadata.json"
        write_jsonl(records, str(input_path))
        print(
            f"Azure batch input ready: {input_path} ({len(records)} records)",
            flush=True,
        )

        client = self._client()
        with input_path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        uploaded_dump = safe_model_dump(uploaded)
        print(
            f"Azure batch file uploaded: file_id={getattr(uploaded, 'id', uploaded_dump.get('id', ''))}",
            flush=True,
        )

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint=endpoint,
            completion_window=str(self.api_settings.get("batch_completion_window", "24h")),
        )
        batch_dump = safe_model_dump(batch)
        print(
            f"Azure batch created: batch_id={getattr(batch, 'id', batch_dump.get('id', ''))}, "
            f"endpoint={endpoint}",
            flush=True,
        )
        meta = {
            "input_file": uploaded_dump,
            "batch_initial": batch_dump,
            "endpoint": endpoint,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(meta, str(meta_path))

        poll_interval = int(self.api_settings.get("batch_poll_interval_seconds", 60) or 60)
        max_wait_seconds = int(self.api_settings.get("batch_max_wait_seconds", 0) or 0)
        t0 = time.time()
        final = batch
        with status_path.open("a", encoding="utf-8") as f:
            while True:
                final = client.batches.retrieve(batch.id)
                final_dump = safe_model_dump(final)
                f.write(json.dumps(final_dump, ensure_ascii=False) + "\n")
                f.flush()
                status = str(getattr(final, "status", "") or final_dump.get("status", "")).lower()
                request_counts = final_dump.get("request_counts") or {}
                total = request_counts.get("total", "?")
                completed = request_counts.get("completed", "?")
                failed = request_counts.get("failed", "?")
                elapsed_min = (time.time() - t0) / 60
                print(
                    "Azure batch status: "
                    f"batch_id={batch.id}, status={status}, "
                    f"completed={completed}/{total}, failed={failed}, "
                    f"elapsed={elapsed_min:.1f}m",
                    flush=True,
                )
                if status in TERMINAL_BATCH_STATUSES:
                    break
                if max_wait_seconds and time.time() - t0 > max_wait_seconds:
                    raise TimeoutError(
                        f"Azure batch {batch.id} exceeded max wait {max_wait_seconds}s; "
                        f"last status={status}"
                    )
                time.sleep(max(10, poll_interval))

        final_dump = safe_model_dump(final)
        meta["batch_final"] = final_dump
        write_json(meta, str(meta_path))
        status = str(final_dump.get("status", "")).lower()
        if status != "completed":
            raise RuntimeError(f"Azure batch {batch.id} ended with status={status}. See {meta_path}")

        output_file_id = final_dump.get("output_file_id")
        error_file_id = final_dump.get("error_file_id")
        if not output_file_id:
            raise RuntimeError(f"Azure batch {batch.id} completed without output_file_id.")
        print(f"Downloading Azure batch output: file_id={output_file_id}", flush=True)
        download_azure_file(client, output_file_id, output_path)
        if error_file_id:
            print(f"Downloading Azure batch errors: file_id={error_file_id}", flush=True)
            download_azure_file(client, error_file_id, error_path)

        parsed = parse_azure_result_batch(str(output_path))
        print(f"Azure batch parsed: {len(parsed)} results", flush=True)
        results = []
        for item in parsed:
            results.append(
                QueryResult(
                    custom_id=item.get("id", ""),
                    text=item.get("output", ""),
                    summary=item.get("summary", ""),
                    error=json.dumps(item.get("error"), ensure_ascii=False)
                    if item.get("error")
                    else "",
                    raw=item,
                )
            )
        if not keep_artifacts:
            for path in (
                input_path,
                output_path,
                error_path,
                status_path,
                meta_path,
            ):
                if path.exists():
                    path.unlink()
            try:
                trace_dir.rmdir()
            except OSError:
                pass
        return results


def download_azure_file(client: Any, file_id: str, path: Path) -> None:
    """Download an Azure/OpenAI file content response to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    response = client.files.content(file_id)
    if hasattr(response, "write_to_file"):
        response.write_to_file(str(path))
        return
    content = None
    if isinstance(response, bytes):
        content = response
    elif isinstance(response, str):
        path.write_text(response, encoding="utf-8")
        return
    elif hasattr(response, "content"):
        content = response.content
    elif hasattr(response, "read"):
        content = response.read()
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif content is not None:
        path.write_text(str(content), encoding="utf-8")
    else:
        path.write_text(str(response), encoding="utf-8")


def parse_azure_result_batch(path_file_result_batch: str) -> list[dict[str, Any]]:
    """Parse Azure batch output JSONL into normalized provider rows."""
    parsed = []
    with Path(path_file_result_batch).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid Azure batch JSONL at line {line_number}: "
                    f"{path_file_result_batch}"
                ) from exc
            custom_id_value = str(item.get("custom_id") or "")
            response = item.get("response") or {}
            body = response.get("body") or {}
            error = item.get("error") or response.get("error")
            output = ""
            summary = ""
            if body:
                output = extract_text_from_responses_api_output(body)
                summary = extract_summary_from_responses_api_output(body)
                if not output:
                    output = extract_text_from_chat_completion_body(body)
            parsed.append(
                {
                    "id": custom_id_value,
                    "output": output,
                    "summary": summary,
                    "error": error,
                    "response": response,
                }
            )
    return parsed
