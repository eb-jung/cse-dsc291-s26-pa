from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


M = 2048
N = 2048
K = 2048
WARMUP_ITERS = 20
BENCHMARK_REPEATS = 500
SEARCH_WARMUP_ITERS = 3
SEARCH_REPEATS = 10
CORRECTNESS_SHAPE = 512
ATOL = 0.15
RTOL = 0.032
KERNEL_FUNCTION_NAME = "matmul_add_relu_kernel_fp16"
CONFIGS_NAME = "KERNEL_CONFIGS"
MIN_CONFIGS = 1
MAX_CONFIGS = 5
EXPECTED_KERNEL_ARGS = [
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "d_ptr",
    "M",
    "N",
    "K",
    "stride_am",
    "stride_ak",
    "stride_bk",
    "stride_bn",
    "stride_cm",
    "stride_cn",
    "stride_dm",
    "stride_dn",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
]
EXPECTED_CONFIG_KEYS = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "num_warps",
    "num_stages",
]


def _is_triton_jit_decorator(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "jit"
        and isinstance(node.value, ast.Name)
        and node.value.id == "triton"
    )


def normalize_configs(configs: Any) -> tuple[bool, str, list[dict[str, int]] | None]:
    if not isinstance(configs, list):
        return False, f"{CONFIGS_NAME} must be a list of dictionaries.", None
    if not MIN_CONFIGS <= len(configs) <= MAX_CONFIGS:
        return (
            False,
            f"{CONFIGS_NAME} must contain between {MIN_CONFIGS} and {MAX_CONFIGS} configs.",
            None,
        )

    normalized: list[dict[str, int]] = []
    expected_keys = set(EXPECTED_CONFIG_KEYS)
    for idx, config in enumerate(configs):
        if not isinstance(config, dict):
            return False, f"{CONFIGS_NAME}[{idx}] must be a dictionary.", None
        if set(config) != expected_keys:
            return (
                False,
                f"{CONFIGS_NAME}[{idx}] must contain exactly these keys: "
                f"{', '.join(EXPECTED_CONFIG_KEYS)}",
                None,
            )

        normalized_config: dict[str, int] = {}
        for key in EXPECTED_CONFIG_KEYS:
            value = config[key]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                return False, f"{CONFIGS_NAME}[{idx}]['{key}'] must be a positive integer.", None
            normalized_config[key] = int(value)
        normalized.append(normalized_config)

    return True, "ok", normalized


def validate_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    if len(tree.body) != 2:
        raise RuntimeError(
            f"Submission must contain exactly two top-level definitions: {CONFIGS_NAME} and "
            f"{KERNEL_FUNCTION_NAME}."
        )

    config_node = None
    kernel_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if config_node is not None:
                raise RuntimeError(f"Submission must define {CONFIGS_NAME} exactly once.")
            config_node = node
        elif isinstance(node, ast.FunctionDef):
            if kernel_node is not None:
                raise RuntimeError(f"Submission must define {KERNEL_FUNCTION_NAME} exactly once.")
            kernel_node = node
        else:
            raise RuntimeError(
                f"Only a top-level {CONFIGS_NAME} assignment and {KERNEL_FUNCTION_NAME} are allowed."
            )

    if config_node is None:
        raise RuntimeError(f"Submission must define top-level {CONFIGS_NAME}.")
    if kernel_node is None:
        raise RuntimeError(f"Submission must define top-level {KERNEL_FUNCTION_NAME}.")

    if len(config_node.targets) != 1:
        raise RuntimeError(f"{CONFIGS_NAME} must be assigned exactly once.")
    if not isinstance(config_node.targets[0], ast.Name) or config_node.targets[0].id != CONFIGS_NAME:
        raise RuntimeError(f"Top-level assignment must be named {CONFIGS_NAME}.")

    try:
        configs_value = ast.literal_eval(config_node.value)
    except Exception as exc:
        raise RuntimeError(f"{CONFIGS_NAME} must be a literal list of dictionaries.") from exc

    ok, message, _ = normalize_configs(configs_value)
    if not ok:
        raise RuntimeError(message)

    if kernel_node.name != KERNEL_FUNCTION_NAME:
        raise RuntimeError(f"Top-level function must be named {KERNEL_FUNCTION_NAME}.")
    if len(kernel_node.decorator_list) != 1 or not _is_triton_jit_decorator(kernel_node.decorator_list[0]):
        raise RuntimeError(f"{KERNEL_FUNCTION_NAME} must be decorated with exactly @triton.jit.")

    arg_names = [arg.arg for arg in kernel_node.args.args]
    if arg_names != EXPECTED_KERNEL_ARGS:
        raise RuntimeError(
            f"{KERNEL_FUNCTION_NAME} must have exactly this signature: "
            f"{', '.join(EXPECTED_KERNEL_ARGS)}"
        )


def load_student_submission(path: Path):
    source = path.read_text(encoding="utf-8")
    namespace = {
        "__builtins__": __builtins__,
        "torch": torch,
        "triton": triton,
        "tl": tl,
    }
    exec(compile(source, str(path), "exec"), namespace, namespace)

    kernel = namespace.get(KERNEL_FUNCTION_NAME)
    if kernel is None or not callable(kernel):
        raise RuntimeError(f"Missing callable {KERNEL_FUNCTION_NAME}")

    raw_configs = namespace.get(CONFIGS_NAME)
    if raw_configs is None:
        raise RuntimeError(f"Missing {CONFIGS_NAME}")

    ok, message, configs = normalize_configs(raw_configs)
    if not ok:
        raise RuntimeError(message)

    return kernel, configs


def build_wrapper(kernel, config: dict[str, int]):
    def wrapped(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        m, k = a.shape
        k2, n = b.shape
        assert k == k2
        assert c.shape == (m, n)
        d = torch.empty((m, n), device=a.device, dtype=torch.float16)
        grid = (triton.cdiv(m, config["BLOCK_M"]), triton.cdiv(n, config["BLOCK_N"]))
        kernel[grid](
            a,
            b,
            c,
            d,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            d.stride(0),
            d.stride(1),
            BLOCK_M=config["BLOCK_M"],
            BLOCK_N=config["BLOCK_N"],
            BLOCK_K=config["BLOCK_K"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return d

    return wrapped


def reference_matmul_add_relu(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b).add(c).relu_()


def time_kernel(
    fn,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    warmup_iters: int,
    repeats: int,
) -> float:
    _ = fn(a, b, c)
    torch.cuda.synchronize()

    for _ in range(warmup_iters):
        _ = fn(a, b, c)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = fn(a, b, c)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0


def select_best_config(kernel, configs: list[dict[str, int]]) -> tuple[dict[str, int], dict]:
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.randn((M, N), device="cuda", dtype=torch.float16)

    results = []
    failures = []
    for config in configs:
        fn = build_wrapper(kernel, config)
        try:
            search_ms = time_kernel(fn, a, b, c, SEARCH_WARMUP_ITERS, SEARCH_REPEATS)
            results.append({"config": config, "search_ms": search_ms})
        except Exception:
            failures.append({"config": config, "traceback": traceback.format_exc()})

    if not results:
        raise RuntimeError("All submitted configs failed during config search.")

    best = min(results, key=lambda item: item["search_ms"])
    return best["config"], {
        "warmup_iters": SEARCH_WARMUP_ITERS,
        "repeats": SEARCH_REPEATS,
        "results": results,
        "failures": failures,
    }


def correctness(fn) -> dict:
    torch.manual_seed(0)
    a = torch.randn((CORRECTNESS_SHAPE, CORRECTNESS_SHAPE), device="cuda", dtype=torch.float16)
    b = torch.randn((CORRECTNESS_SHAPE, CORRECTNESS_SHAPE), device="cuda", dtype=torch.float16)
    c = torch.randn((CORRECTNESS_SHAPE, CORRECTNESS_SHAPE), device="cuda", dtype=torch.float16)

    out = fn(a, b, c)
    ref = reference_matmul_add_relu(a, b, c)
    max_abs_diff = float(torch.max(torch.abs(out - ref)).item())
    ok = torch.allclose(out, ref, atol=ATOL, rtol=RTOL)

    return {
        "ok": bool(ok),
        "max_abs_diff": max_abs_diff,
        "output_dtype": str(out.dtype),
    }


def benchmark(fn) -> dict:
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.randn((M, N), device="cuda", dtype=torch.float16)

    student_ms = time_kernel(fn, a, b, c, WARMUP_ITERS, BENCHMARK_REPEATS)

    t0 = time.perf_counter()
    for _ in range(BENCHMARK_REPEATS):
        _ = reference_matmul_add_relu(a, b, c)
    torch.cuda.synchronize()
    reference_ms = (time.perf_counter() - t0) / BENCHMARK_REPEATS * 1000.0

    return {
        "student_ms": student_ms,
        "reference_ms": reference_ms,
        "speedup_vs_pytorch": reference_ms / student_ms,
        "benchmark_shape": [M, N, K],
        "warmup_iters": WARMUP_ITERS,
        "repeats": BENCHMARK_REPEATS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local correctness and speed test for student_kernel.py"
    )
    parser.add_argument("submission", help="Path to student_kernel.py")
    args = parser.parse_args()

    path = Path(args.submission).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Submission file not found: {path}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this test script.")

    try:
        validate_source(path)
        kernel, configs = load_student_submission(path)
        selected_config, config_search = select_best_config(kernel, configs)
        fn = build_wrapper(kernel, selected_config)
    except Exception:
        print(traceback.format_exc())
        raise SystemExit("Submission validation or config search failed.")

    result = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "submitted_configs": configs,
        "selected_config": selected_config,
        "config_search": config_search,
    }

    try:
        result["correctness"] = correctness(fn)
    except Exception:
        print(traceback.format_exc())
        raise SystemExit("Correctness check failed with an exception.")

    if not result["correctness"]["ok"]:
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit("Correctness check failed.")

    try:
        result["benchmark"] = benchmark(fn)
    except Exception:
        print(traceback.format_exc())
        raise SystemExit("Benchmark failed with an exception.")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
