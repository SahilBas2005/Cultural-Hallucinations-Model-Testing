import json
import math
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

_LLAMA_CPP_TAG = "b11982"
_LLAMA_CPP_BIN_ZIP = "llama-b11982-bin-win-cuda-12.4-x64.zip"
_LLAMA_CPP_CUDART_ZIP = "cudart-llama-bin-win-cuda-12.4-x64.zip"
_LLAMA_CPP_RELEASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{_LLAMA_CPP_TAG}"
_LLAMA_CPP_ROOT = Path.home() / ".cache" / "navarasa" / "llama.cpp"
_GPU_OFFLOAD_RE = re.compile(r"offloaded (\d+)/(\d+) layers to GPU", re.IGNORECASE)
TOP_LOGPROBS = 5


def _test_model_attr(name: str):
    # `python test_model.py` loads that file as __main__, so import it only as a fallback.
    main = sys.modules.get("__main__")
    main_file = getattr(main, "__file__", None)
    if (
        main is not None
        and main_file
        and Path(main_file).name == "test_model.py"
        and hasattr(main, name)
    ):
        return getattr(main, name)

    module = sys.modules.get("test_model")
    if module is None:
        import test_model as module
    return getattr(module, name)


def _download_file(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "llama-cpp-local"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial.replace(destination)


def _llama_cpp_dir() -> Path:
    directory = _LLAMA_CPP_ROOT / _LLAMA_CPP_TAG
    server = directory / "llama-server.exe"
    runtime = list(directory.glob("cudart64_*.dll"))
    if server.is_file() and runtime:
        return directory

    directory.mkdir(parents=True, exist_ok=True)
    for name in (_LLAMA_CPP_BIN_ZIP, _LLAMA_CPP_CUDART_ZIP):
        archive = directory / name
        _download_file(f"{_LLAMA_CPP_RELEASE}/{name}", archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(directory)
    if not server.is_file() or not list(directory.glob("cudart64_*.dll")):
        raise RuntimeError(f"CUDA llama.cpp build is incomplete in {directory}")
    return directory


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _require_full_gpu_offload(log_text: str) -> None:
    match = _GPU_OFFLOAD_RE.search(log_text)
    if match is None:
        raise RuntimeError(
            "llama.cpp did not report GPU layer offload. "
            "The model must run entirely on the GPU.\n"
            + log_text[-4000:]
        )
    used = int(match.group(1))
    total = int(match.group(2))
    if used == 0 or used != total:
        raise RuntimeError(
            f"llama.cpp offloaded {used}/{total} layers to the GPU. "
            "Expected every layer on the GPU.\n"
            + log_text[-4000:]
        )
    cpu_buffer = re.search(r"CPU_Mapped model buffer size =\s+([0-9.]+) MiB", log_text)
    if cpu_buffer and float(cpu_buffer.group(1)) > 1:
        raise RuntimeError(
            "llama.cpp left "
            f"{cpu_buffer.group(1)} MiB of model weights on the CPU.\n"
            + log_text[-4000:]
        )


def _wait_for_server(
    process: subprocess.Popen,
    url: str,
    log_path: Path,
    timeout: float = 180,
    require_full_gpu: bool = True,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "llama-server exited before the model finished loading.\n"
                + _read_text(log_path)[-4000:]
                + "\n"
                + _read_text(log_path.with_suffix(".stderr"))[-4000:]
            )
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                body = json.load(response)
            if body.get("status") == "ok":
                if require_full_gpu:
                    _require_full_gpu_offload(_read_text(log_path))
                return
        except urllib.error.HTTPError:
            pass
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            pass
        time.sleep(0.5)
    raise RuntimeError(
        "llama-server did not become ready.\n" + _read_text(log_path)[-4000:]
    )


def _llama_post(url: str, path: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama.cpp {path} failed ({error.code}): {detail}") from error
    return body


def _number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _logprob_and_probability(item: dict) -> tuple[float | None, float | None]:
    logprob = _number(item.get("logprob"))
    probability = None
    if logprob is not None:
        try:
            probability = math.exp(logprob)
        except OverflowError:
            probability = None
        logprob = round(logprob, 6)
    if probability is not None:
        probability = round(probability, 6)
    return logprob, probability


def _token_record(item) -> dict:
    if not isinstance(item, dict):
        item = {}
    logprob, probability = _logprob_and_probability(item)
    record = {"token": item.get("token"), "logprob": logprob, "probability": probability}
    if item.get("id") is not None:
        record["id"] = item.get("id")
    return record


def _generated_tokens(logprobs) -> list[dict]:
    if not isinstance(logprobs, dict):
        return []
    content = logprobs.get("content")
    if not isinstance(content, list):
        return []
    rows = []
    for item in content:
        if not isinstance(item, dict):
            continue
        record = _token_record(item)
        alternatives = item.get("top_logprobs") or []
        if not isinstance(alternatives, list):
            alternatives = []
        record["top"] = [_token_record(top) for top in alternatives]
        rows.append(record)
    return rows


def score_columns(logprobs) -> tuple[str, str]:
    logits = []
    probabilities = []
    for row in _generated_tokens(logprobs):
        logits.append(
            {
                "token": row["token"],
                **({"id": row["id"]} if "id" in row else {}),
                "logprob": row["logprob"],
                "top_logprobs": [
                    {
                        "token": top["token"],
                        **({"id": top["id"]} if "id" in top else {}),
                        "logprob": top["logprob"],
                    }
                    for top in row["top"]
                ],
            }
        )
        probabilities.append(
            {
                "token": row["token"],
                **({"id": row["id"]} if "id" in row else {}),
                "probability": row["probability"],
                "top_probabilities": [
                    {
                        "token": top["token"],
                        **({"id": top["id"]} if "id" in top else {}),
                        "probability": top["probability"],
                    }
                    for top in row["top"]
                ],
            }
        )
    return (
        json.dumps(logits, ensure_ascii=False),
        json.dumps(probabilities, ensure_ascii=False),
    )


def hidden_state_column(vector) -> str:
    if vector and isinstance(vector[0], list):
        vector = vector[-1]
    rounded = []
    for value in vector:
        number = _number(value)
        rounded.append(None if number is None else round(number, 6))
    return json.dumps(rounded, ensure_ascii=False)


def _final_hidden_state(url: str, messages: list[dict], completion: str, timeout: float) -> str:
    rendered = _llama_post(url, "/apply-template", {"messages": messages}, timeout)
    prompt = rendered.get("prompt") or ""
    if not isinstance(prompt, str):
        prompt = json.dumps(prompt, ensure_ascii=False)
    embedded = _llama_post(
        url,
        "/v1/embeddings",
        {"input": f"{prompt}{completion}", "embd_normalize": -1},
        timeout,
    )
    vector = embedded["data"][0]["embedding"]
    return hidden_state_column(vector)


def _max_tokens(external_CoT: bool) -> int:
    return 4096 if external_CoT else 256


def _completion_timeout(external_CoT: bool) -> int:
    return 900 if external_CoT else 180


def _generation_record(url: str, messages: list[dict], choice: dict, timeout: float) -> dict:
    content = (choice.get("message") or {}).get("content") or ""
    logits, probabilities = score_columns((choice.get("logprobs") or {}))
    return {
        "text": content,
        "logits": logits,
        "token_probabilities": probabilities,
        "hidden_states": _final_hidden_state(url, messages, content, timeout),
    }
