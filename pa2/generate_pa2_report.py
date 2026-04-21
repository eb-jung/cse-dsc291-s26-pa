from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
JSON_REPORT = ROOT / "pa2_report.json"
DISCUSSION_FILE = ROOT / "discussion2-1.txt"

PART2_PYTEST_SPECS = {
    "test_data_split": {"section": "2.2", "points": 5.0, "expected_tests": 4},
    "test_get_info": {"section": "2.3", "points": 15.0, "expected_tests": 2},
    "test_transformer_forward": {"section": "2.4", "points": 10.0, "expected_tests": 2},
    "test_transformer_backward": {"section": "2.5", "points": 10.0, "expected_tests": 2},
}

TEST_STATUS_RE = re.compile(r"(tests/[^\s:]+::[^\s]+)\s+(PASSED|FAILED|ERROR|SKIPPED|XPASSED|XFAILED)")
TIME_RE = re.compile(r"Average\s+(.+?)\s+time:\s+([0-9]*\.?[0-9]+)\s+seconds")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SUMMARY_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xpassed|xfailed)")


def maybe_parse_json(text: str) -> dict | list | None:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def run_command(name: str, command: list[str]) -> dict:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    result = {
        "name": name,
        "command": command,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    parsed_stdout = maybe_parse_json(proc.stdout)
    if parsed_stdout is not None:
        result["parsed_stdout"] = parsed_stdout
    return result


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def load_discussion_summary() -> dict:
    if not DISCUSSION_FILE.exists():
        return {"exists": False, "nonempty": False, "path": str(DISCUSSION_FILE)}
    text = DISCUSSION_FILE.read_text(encoding="utf-8").strip()
    return {
        "exists": True,
        "nonempty": bool(text),
        "path": str(DISCUSSION_FILE),
        "chars": len(text),
    }


def extract_test_statuses(text: str) -> dict[str, str]:
    text = ANSI_ESCAPE_RE.sub("", text).replace("\r", "\n")
    rank = {"ERROR": 4, "FAILED": 3, "XPASSED": 2, "PASSED": 1, "SKIPPED": 0, "XFAILED": 0}
    statuses: dict[str, str] = {}
    for test_name, status in TEST_STATUS_RE.findall(text):
        prev = statuses.get(test_name)
        if prev is None or rank[status] > rank[prev]:
            statuses[test_name] = status
    return statuses


def parse_pytest_summary_counts(text: str) -> dict[str, int]:
    cleaned = ANSI_ESCAPE_RE.sub("", text).replace("\r", "\n")
    counts = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "xpassed": 0,
        "xfailed": 0,
    }
    for line in cleaned.splitlines():
        if " in " not in line:
            continue
        local = dict(counts)
        for value, label in SUMMARY_COUNT_RE.findall(line):
            value_int = int(value)
            if label == "passed":
                local["passed"] = value_int
            elif label == "failed":
                local["failed"] = value_int
            elif label in {"error", "errors"}:
                local["errors"] = value_int
            elif label == "skipped":
                local["skipped"] = value_int
            elif label == "xpassed":
                local["xpassed"] = value_int
            elif label == "xfailed":
                local["xfailed"] = value_int
        if any(local.values()):
            for key, value in local.items():
                counts[key] = max(counts[key], value)
    return counts


def score_pytest_result(result: dict, spec: dict) -> dict:
    combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    expected_tests = spec["expected_tests"]
    if result.get("returncode") == 0:
        passed = expected_tests
        failed = 0
        parse_method = "returncode"
        summary_counts = {"passed": expected_tests, "failed": 0, "errors": 0, "skipped": 0, "xpassed": 0, "xfailed": 0}
        statuses = {}
    else:
        summary_counts = parse_pytest_summary_counts(combined)
        statuses = extract_test_statuses(combined)
        if summary_counts["passed"] or summary_counts["failed"] or summary_counts["errors"]:
            passed = min(summary_counts["passed"] + summary_counts["xpassed"], expected_tests)
            failed = min(summary_counts["failed"] + summary_counts["errors"], expected_tests)
            parse_method = "summary_line"
        else:
            passed = min(sum(1 for status in statuses.values() if status in {"PASSED", "XPASSED"}), expected_tests)
            failed = min(sum(1 for status in statuses.values() if status in {"FAILED", "ERROR"}), expected_tests)
            parse_method = "test_status_lines"
    points = spec["points"] * passed / expected_tests
    return {
        "section": spec["section"],
        "points_earned": points,
        "points_max": spec["points"],
        "expected_tests": expected_tests,
        "passed_tests": passed,
        "failed_tests": failed,
        "observed_tests": max(passed + failed, len(statuses)),
        "all_passed": passed == expected_tests and result.get("returncode") == 0,
        "parse_method": parse_method,
        "summary_counts": summary_counts,
        "test_statuses": dict(sorted(statuses.items())),
    }


def parse_mpi_benchmark_times(text: str) -> dict[str, float]:
    times: dict[str, float] = {}
    for label, value in TIME_RE.findall(text):
        times[label] = float(value)
    return times


def score_collective_result(result: dict, section_name: str) -> dict:
    stdout = result.get("stdout", "")
    all_correct = "All runs produced correct results." in stdout and result.get("returncode") == 0
    times = parse_mpi_benchmark_times(stdout)

    if section_name == "myallreduce":
        ref_key = "MPI.Allreduce"
        mine_key = "myAllreduce"
    else:
        ref_key = "MPI.Alltoall"
        mine_key = "myAlltoall"

    ref_time = times.get(ref_key)
    mine_time = times.get(mine_key)
    ratio = None
    bonus = 0.0

    if all_correct and ref_time and mine_time:
        ratio = mine_time / ref_time
        if ratio <= 1.05:
            bonus = 5.0
        elif ratio <= 1.5:
            bonus = 2.5

    return {
        "section": "2.1",
        "points_earned": 10.0 if all_correct else 0.0,
        "points_max": 10.0,
        "bonus_earned": bonus,
        "bonus_max": 5.0,
        "all_correct": all_correct,
        "reference_time_seconds": ref_time,
        "student_time_seconds": mine_time,
        "student_vs_reference_ratio": ratio,
    }


def summarize_part1(result: dict) -> dict:
    parsed = result.get("parsed_stdout")
    if not isinstance(parsed, dict):
        return {
            "score_points": None,
            "score_note": "Part 1 output was not parseable. No local score estimate available.",
        }

    remote = parsed.get("result")
    if not isinstance(remote, dict):
        return {
            "score_points": None,
            "score_note": "Part 1 JSON did not contain a result payload. No local score estimate available.",
        }

    correctness = remote.get("correctness", {})
    correctness_ok = bool(isinstance(correctness, dict) and correctness.get("ok"))
    speedup = remote.get("speedup_vs_pytorch")
    raw_score_points = 0.0
    score_tier = "incorrect"

    if correctness_ok:
        raw_score_points = 20.0
        score_tier = "correct_below_1.0x"
        if isinstance(speedup, (int, float)):
            speedup = float(speedup)
            if speedup >= 1.4:
                raw_score_points = 50.0
                score_tier = ">=1.4x"
            elif speedup >= 1.25:
                raw_score_points = 45.0
                score_tier = ">=1.25x"
            elif speedup >= 1.1:
                raw_score_points = 40.0
                score_tier = ">=1.1x"
            elif speedup >= 1.0:
                raw_score_points = 35.0
                score_tier = ">=1.0x"

    base_points = min(raw_score_points, 40.0)
    extra_credit_points = max(raw_score_points - 40.0, 0.0)

    return {
        "correctness_ok": correctness_ok,
        "max_abs_diff": correctness.get("max_abs_diff") if isinstance(correctness, dict) else None,
        "student_ms": remote.get("student_ms"),
        "reference_ms": remote.get("reference_ms"),
        "speedup_vs_pytorch": remote.get("speedup_vs_pytorch"),
        "device_name": remote.get("device_name"),
        "score_tier": score_tier,
        "base_points": base_points,
        "base_points_max": 40.0,
        "extra_credit_points": extra_credit_points,
        "extra_credit_max": 10.0,
        "total_points_with_extra_credit": raw_score_points,
        "total_points_with_extra_credit_max": 50.0,
        "score_note": (
            "Part 1 rubric: incorrect=0, correct but <1.0x=20, >=1.0x=35, >=1.1x=40, "
            ">=1.25x=45, >=1.4x=50."
        ),
    }


def add_score_summary(report: dict) -> None:
    part2_scores: dict[str, dict] = {}
    for name, spec in PART2_PYTEST_SPECS.items():
        part2_scores[name] = score_pytest_result(report["part2"][name], spec)

    part2_scores["myallreduce"] = score_collective_result(report["part2"]["myallreduce"], "myallreduce")
    part2_scores["myalltoall"] = score_collective_result(report["part2"]["myalltoall"], "myalltoall")

    part2_base_points = sum(item["points_earned"] for item in part2_scores.values())
    part2_base_max = sum(item["points_max"] for item in part2_scores.values())
    part2_bonus_points = sum(item.get("bonus_earned", 0.0) for item in part2_scores.values())
    part2_bonus_max = sum(item.get("bonus_max", 0.0) for item in part2_scores.values())
    part1_summary = summarize_part1(report["part1"])
    part1_base_points = part1_summary.get("base_points", 0.0) or 0.0
    part1_base_max = part1_summary.get("base_points_max", 40.0) or 40.0
    part1_extra_points = part1_summary.get("extra_credit_points", 0.0) or 0.0
    part1_extra_max = part1_summary.get("extra_credit_max", 10.0) or 10.0

    report["summary"] = {
        "part1": part1_summary,
        "part2": {
            "discussion2_1": load_discussion_summary(),
            "section_scores": part2_scores,
            "base_points_earned": part2_base_points,
            "base_points_max": part2_base_max,
            "bonus_points_earned": part2_bonus_points,
            "bonus_points_max": part2_bonus_max,
            "total_points_with_bonus": part2_base_points + part2_bonus_points,
            "total_points_with_bonus_max": part2_base_max + part2_bonus_max,
        },
        "overall": {
            "base_points_earned": part1_base_points + part2_base_points,
            "base_points_max": part1_base_max + part2_base_max,
            "extra_credit_points_earned": part1_extra_points + part2_bonus_points,
            "extra_credit_points_max": part1_extra_max + part2_bonus_max,
            "total_points_with_extra_credit": part1_base_points + part2_base_points + part1_extra_points + part2_bonus_points,
            "total_points_with_extra_credit_max": part1_base_max + part2_base_max + part1_extra_max + part2_bonus_max,
        },
    }


def build_report() -> dict:
    require_env("GRADER_BASE_URL")
    require_env("GRADER_TOKEN")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(ROOT),
        "part1": {},
        "part2": {},
    }

    part1_result = run_command(
        "part1_remote_submit",
        [sys.executable, "student_submit.py", "--json", "student_kernel.py"],
    )
    report["part1"] = part1_result

    part2_commands = [
        (
            "myallreduce",
            ["mpirun", "-n", "8", sys.executable, "mpi-test.py", "--test_case", "myallreduce"],
        ),
        (
            "myalltoall",
            ["mpirun", "-n", "8", sys.executable, "mpi-test.py", "--test_case", "myalltoall"],
        ),
        (
            "test_data_split",
            [sys.executable, "-m", "pytest", "-l", "-v", "tests/test_data_split.py"],
        ),
        (
            "test_get_info",
            ["mpirun", "-n", "8", sys.executable, "-m", "pytest", "-l", "-v", "--with-mpi", "tests/test_get_info.py"],
        ),
        (
            "test_transformer_forward",
            ["mpirun", "-n", "4", sys.executable, "-m", "pytest", "-l", "-v", "--with-mpi", "tests/test_transformer_forward.py"],
        ),
        (
            "test_transformer_backward",
            ["mpirun", "-n", "4", sys.executable, "-m", "pytest", "-l", "-v", "--with-mpi", "tests/test_transformer_backward.py"],
        ),
    ]

    for name, command in part2_commands:
        report["part2"][name] = run_command(name, command)

    add_score_summary(report)
    return report


def format_console_summary(report: dict) -> str:
    lines: list[str] = []
    lines.append("PA2 Report")
    lines.append(f"Generated at (UTC): {report['generated_at_utc']}")
    lines.append("")
    lines.append("Summary")

    part1_summary = report.get("summary", {}).get("part1", {})
    lines.append("Part 1")
    correctness_text = "yes" if part1_summary.get("correctness_ok") else "no"
    if part1_summary.get("correctness_ok") is not None:
        lines.append(f"Correctness passed: {correctness_text}")
    if part1_summary.get("student_ms") is not None:
        lines.append(f"Your kernel: {part1_summary['student_ms']:.4f} ms")
    if part1_summary.get("reference_ms") is not None:
        lines.append(f"PyTorch ref: {part1_summary['reference_ms']:.4f} ms")
    if part1_summary.get("speedup_vs_pytorch") is not None:
        lines.append(f"Speedup vs ref: {part1_summary['speedup_vs_pytorch']:.4f}x")
    if part1_summary.get("base_points") is not None:
        lines.append(f"Part 1 base score: {part1_summary['base_points']:.1f}/{part1_summary.get('base_points_max', 40.0):.1f}")
    if part1_summary.get("extra_credit_points") is not None:
        lines.append(
            f"Part 1 extra credit: {part1_summary['extra_credit_points']:.1f}/"
            f"{part1_summary.get('extra_credit_max', 10.0):.1f}"
        )
    if part1_summary.get("total_points_with_extra_credit") is not None:
        lines.append(
            f"Part 1 total with extra credit: {part1_summary['total_points_with_extra_credit']:.1f}/"
            f"{part1_summary.get('total_points_with_extra_credit_max', 50.0):.1f}"
        )
    if part1_summary.get("score_tier"):
        lines.append(f"Part 1 tier: {part1_summary['score_tier']}")
    lines.append(f"Scoring: {part1_summary.get('score_note', 'Unavailable')}")
    lines.append("")

    part2_summary = report.get("summary", {}).get("part2", {})
    lines.append("Part 2")
    for key in ["myallreduce", "myalltoall", "test_data_split", "test_get_info", "test_transformer_forward", "test_transformer_backward"]:
        item = part2_summary.get("section_scores", {}).get(key)
        if not item:
            continue
        bonus_text = ""
        if item.get("bonus_max"):
            bonus_text = f" + bonus {item.get('bonus_earned', 0.0):.1f}/{item['bonus_max']:.1f}"
        lines.append(
            f"{key}: {item['points_earned']:.1f}/{item['points_max']:.1f}{bonus_text}"
        )
    discussion = part2_summary.get("discussion2_1", {})
    lines.append(
        f"discussion2-1.txt present: {'yes' if discussion.get('exists') else 'no'}, "
        f"nonempty: {'yes' if discussion.get('nonempty') else 'no'}"
    )
    lines.append(
        f"Part 2 base score: {part2_summary.get('base_points_earned', 0.0):.1f}/{part2_summary.get('base_points_max', 0.0):.1f}"
    )
    lines.append(
        f"Part 2 bonus: {part2_summary.get('bonus_points_earned', 0.0):.1f}/{part2_summary.get('bonus_points_max', 0.0):.1f}"
    )
    lines.append(
        f"Part 2 total with bonus: {part2_summary.get('total_points_with_bonus', 0.0):.1f}/{part2_summary.get('total_points_with_bonus_max', 0.0):.1f}"
    )
    lines.append("")

    overall = report.get("summary", {}).get("overall", {})
    if overall:
        lines.append("Overall")
        lines.append(
            f"Base score: {overall.get('base_points_earned', 0.0):.1f}/{overall.get('base_points_max', 100.0):.1f}"
        )
        lines.append(
            f"Extra credit: {overall.get('extra_credit_points_earned', 0.0):.1f}/"
            f"{overall.get('extra_credit_points_max', 20.0):.1f}"
        )
        lines.append(
            "Total with extra credit: "
            f"{overall.get('total_points_with_extra_credit', 0.0):.1f}/"
            f"{overall.get('total_points_with_extra_credit_max', 120.0):.1f}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    report = build_report()
    console_summary = format_console_summary(report)
    report["console_summary"] = console_summary
    report["artifacts"] = {
        "json_report": str(JSON_REPORT),
    }
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(console_summary, end="")


if __name__ == "__main__":
    main()
