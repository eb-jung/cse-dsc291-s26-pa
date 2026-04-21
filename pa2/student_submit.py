from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def _write_output(path_str: str | None, text: str) -> None:
    if not path_str:
        return
    Path(path_str).expanduser().write_text(text, encoding="utf-8")


def _print_submit_summary(payload: dict) -> None:
    print("Submission accepted.")
    print(f"Call ID: {payload['call_id']}")
    print(f"Filename: {payload['filename']}")
    print(f"Size: {payload['file_size_bytes']} bytes")


def _format_config(config: dict | None) -> str | None:
    if not isinstance(config, dict):
        return None
    ordered_keys = ["BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages"]
    parts = []
    for key in ordered_keys:
        if key in config:
            parts.append(f"{key}={config[key]}")
    if not parts:
        return None
    return ", ".join(parts)


def _print_waiting_status(call_id: str, attempt: int) -> None:
    dots = "." * ((attempt % 3) + 1)
    message = f"\rWaiting for result{dots:<3} call_id={call_id}"
    print(message, end="", flush=True)


def _extract_runner_payload(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _merged_error_payload(payload: dict) -> dict:
    stdout = payload.get("stdout", "")
    nested = _extract_runner_payload(stdout) if stdout else None
    if not nested:
        return payload

    merged = dict(payload)
    merged.update(nested)
    merged["stdout"] = payload.get("stdout", "")
    merged["stderr"] = payload.get("stderr", "")
    return merged


def _last_error_line(traceback_text: str) -> str | None:
    for line in reversed(traceback_text.splitlines()):
        stripped = line.strip()
        if not stripped or stripped == "^":
            continue
        if stripped.startswith("Traceback "):
            continue
        return stripped
    return None


def _extract_compilation_block(traceback_text: str) -> str | None:
    lines = traceback_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "CompilationError:" in line:
            start = i
            break

    if start is None:
        return None

    block = "\n".join(lines[start:]).strip()
    return block or None


def _friendly_hint(error_payload: dict) -> str | None:
    combined = "\n".join(
        str(part)
        for part in (
            error_payload.get("message", ""),
            error_payload.get("traceback", ""),
            error_payload.get("stderr", ""),
            error_payload.get("stdout", ""),
        )
        if part
    )

    if "zeros() missing 1 required positional argument: 'dtype'" in combined:
        return (
            "`tl.zeros` needs an explicit dtype. For the accumulator, write "
            "`tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)`."
        )

    if "Top-level function must be named" in combined:
        return "Check that your file only defines `matmul_add_relu_kernel_fp16`."

    if "must be decorated with exactly @triton.jit" in combined:
        return "Add exactly one decorator: `@triton.jit`."

    if "must have exactly this signature" in combined:
        return "Match the required kernel argument list exactly, including argument order."

    if "Submission must contain exactly two top-level definitions" in combined:
        return "Your file should contain only `KERNEL_CONFIGS = [...]` and `matmul_add_relu_kernel_fp16`."

    if "must define top-level KERNEL_CONFIGS" in combined or "must define KERNEL_CONFIGS" in combined:
        return "Add a top-level `KERNEL_CONFIGS = [...]` list before the kernel."

    if "KERNEL_CONFIGS must contain between" in combined:
        return "Submit a short candidate list, e.g. 1 to 5 configs."

    if "must contain exactly these keys" in combined:
        return "Each config must include BLOCK_M, BLOCK_N, BLOCK_K, num_warps, and num_stages."

    return None


def _print_block(title: str, body: str) -> None:
    print(f"\n{title}")
    print(body.rstrip())


def _print_http_error(response: requests.Response) -> None:
    try:
        payload = response.json()
    except ValueError:
        print(response.text)
        return

    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message")
        if detail:
            print(f"Error: {detail}")
            return

    print(json.dumps(payload, indent=2, sort_keys=True))


def _http_error_payload(response: requests.Response) -> dict:
    payload: dict[str, object] = {
        "http_status": response.status_code,
    }
    try:
        body = response.json()
    except ValueError:
        payload["body"] = response.text
        return payload

    if isinstance(body, dict):
        payload.update(body)
    else:
        payload["body"] = body
    return payload


def _print_result_summary(payload: dict) -> None:
    status = payload.get("status", "unknown")
    print(f"Status: {status}")

    if status != "ok":
        error_payload = _merged_error_payload(payload)

        message = error_payload.get("message")
        if message:
            print(f"Message: {message}")

        correctness = error_payload.get("correctness")
        if isinstance(correctness, dict):
            print(f"Passed correctness: {'yes' if correctness.get('ok') else 'no'}")
            if correctness.get("max_abs_diff") is not None:
                print(f"Max abs diff: {correctness['max_abs_diff']}")

        traceback_text = error_payload.get("traceback", "")
        if traceback_text:
            main_error = _last_error_line(traceback_text)
            if main_error:
                print(f"Main error: {main_error}")

            compilation_block = _extract_compilation_block(traceback_text)
            if compilation_block:
                _print_block("Triton compiler output:", compilation_block)

        hint = _friendly_hint(error_payload)
        if hint:
            print(f"Likely fix: {hint}")

        stderr = error_payload.get("stderr", "")
        if stderr:
            _print_block("stderr:", stderr)

        if not traceback_text:
            stdout = error_payload.get("stdout", "")
            if stdout:
                nested = _extract_runner_payload(stdout)
                if not nested:
                    _print_block("stdout:", stdout)

        print("Use `--json` if you want the full raw payload.")
        return

    correctness = payload.get("correctness", {})
    passed = bool(correctness.get("ok"))
    print(f"Passed correctness: {'yes' if passed else 'no'}")

    if correctness.get("max_abs_diff") is not None:
        print(f"Max abs diff: {correctness['max_abs_diff']:.6f}")

    selected_config = payload.get("selected_config")
    selected_config_text = _format_config(selected_config)
    if selected_config_text:
        print(f"Selected config: {selected_config_text}")

    submitted_configs = payload.get("submitted_configs")
    if isinstance(submitted_configs, list):
        print(f"Submitted configs: {len(submitted_configs)}")

    student_ms = payload.get("student_ms")
    reference_ms = payload.get("reference_ms")
    speedup = payload.get("speedup_vs_pytorch")

    if student_ms is not None:
        print(f"Your kernel: {student_ms:.4f} ms")
    if reference_ms is not None:
        print(f"PyTorch ref: {reference_ms:.4f} ms")
    if speedup is not None:
        print(f"Speedup vs ref: {speedup:.4f}x")

    device_name = payload.get("device_name")
    if device_name:
        print(f"Device: {device_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a Triton submission to the course Modal grader and poll for the result."
    )
    parser.add_argument("submission", help="Path to student_kernel.py")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GRADER_BASE_URL", "").rstrip("/"),
        help="Base URL for the deployed grader, e.g. https://<workspace>--web-app.modal.run",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GRADER_TOKEN", ""),
        help="Course grader token. Defaults to GRADER_TOKEN.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between polling attempts.",
    )
    parser.add_argument(
        "--submit-timeout",
        type=float,
        default=180.0,
        help="Read timeout in seconds for the initial /submit request.",
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=60.0,
        help="Read timeout in seconds for each /result poll request.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional file to write the final output payload to.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    output_group.add_argument(
        "--human",
        dest="json_output",
        action="store_false",
        help="Print a human-readable summary. This is the default.",
    )
    parser.set_defaults(json_output=False)
    args = parser.parse_args()

    submission_path = Path(args.submission).expanduser().resolve()
    if not submission_path.exists():
        raise SystemExit(f"Submission file not found: {submission_path}")
    if submission_path.suffix != ".py":
        raise SystemExit("Submission must be a .py file")
    if not args.base_url:
        raise SystemExit("Provide --base-url or set GRADER_BASE_URL")
    if not args.token:
        raise SystemExit("Provide --token or set GRADER_TOKEN")

    headers = {
        "X-CSE291-DSC291-Token": args.token,
    }

    warmup_payload: dict[str, object]

    # Best-effort warmup to reduce first-request cold start latency.
    try:
        warmup = requests.get(
            f"{args.base_url}/healthz",
            headers=headers,
            timeout=(10, 60),
        )
        warmup_payload = {"ok": True, "http_status": warmup.status_code}
        if not args.json_output:
            print(f"[warmup] /healthz -> HTTP {warmup.status_code}")
    except requests.RequestException as exc:
        warmup_payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        if not args.json_output:
            print(f"[warmup] skipped: {type(exc).__name__}: {exc}")

    with submission_path.open("rb") as f:
        response = requests.post(
            f"{args.base_url}/submit",
            headers=headers,
            files={"file": (submission_path.name, f, "text/x-python")},
            timeout=(10, args.submit_timeout),
        )

    if response.status_code >= 400:
        if args.json_output:
            output_text = json.dumps(
                {
                    "status": "submit_http_error",
                    "warmup": warmup_payload,
                    "submit_error": _http_error_payload(response),
                },
                indent=2,
                sort_keys=True,
            )
            _write_output(args.output, output_text + "\n")
            print(output_text)
        else:
            _print_http_error(response)
        raise SystemExit(f"Submit failed with HTTP {response.status_code}")

    submit_payload = response.json()
    call_id = submit_payload["call_id"]
    if not args.json_output:
        _print_submit_summary(submit_payload)

    poll_attempt = 0
    while True:
        poll_response = requests.get(
            f"{args.base_url}/result/{call_id}",
            headers=headers,
            timeout=(10, args.poll_timeout),
        )
        if poll_response.status_code == 202:
            if not args.json_output:
                _print_waiting_status(call_id, poll_attempt)
            poll_attempt += 1
            time.sleep(args.poll_interval)
            continue

        if poll_response.status_code >= 400:
            if not args.json_output and poll_attempt > 0:
                print()
            if args.json_output:
                output_text = json.dumps(
                    {
                        "status": "result_http_error",
                        "warmup": warmup_payload,
                        "submit": submit_payload,
                        "result_error": _http_error_payload(poll_response),
                    },
                    indent=2,
                    sort_keys=True,
                )
                _write_output(args.output, output_text + "\n")
                print(output_text)
            else:
                _print_http_error(poll_response)
            raise SystemExit(f"Polling failed with HTTP {poll_response.status_code}")

        result_payload = poll_response.json()
        if args.json_output:
            output_text = json.dumps(
                {
                    "status": "ok",
                    "warmup": warmup_payload,
                    "submit": submit_payload,
                    "result": result_payload,
                },
                indent=2,
                sort_keys=True,
            )
            _write_output(args.output, output_text + "\n")
            print(output_text)
        else:
            if poll_attempt > 0:
                print()
            print()
            from io import StringIO
            from contextlib import redirect_stdout

            buf = StringIO()
            with redirect_stdout(buf):
                _print_result_summary(result_payload)
            output_text = buf.getvalue()
            _write_output(args.output, output_text)
            print(output_text, end="")
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
