import atexit
import re
import subprocess
import time
from pathlib import Path

from llama_server import (
    _GPU_OFFLOAD_RE,
    _completion_timeout,
    _free_port,
    _generation_record,
    _llama_cpp_dir,
    _llama_post,
    _max_tokens,
    _read_text,
    _test_model_attr,
    _wait_for_server,
    TOP_LOGPROBS,
)

_SARVAM_REPO = "sarvamai/sarvam-30b-gguf"
_SARVAM_CTX = 8192
_SARVAM_CACHE = Path.home() / ".cache" / "sarvam30b"
_THINK_PREFIX_RE = re.compile(r"^\s*<think>\s*</think>\s*")
_sarvam_server_state: dict = {"process": None, "url": None}


def _sarvam_model_path() -> str:
    from huggingface_hub import hf_hub_download

    first = None
    for index in range(1, 7):
        filename = f"sarvam-30b-Q4_K_M.gguf-{index:05d}-of-00006.gguf"
        last_error = None
        for attempt in range(1, 6):
            try:
                path = hf_hub_download(repo_id=_SARVAM_REPO, filename=filename)
                break
            except Exception as error:
                last_error = error
                print(f"retry {attempt}/5 for {filename}: {error}", flush=True)
                time.sleep(5)
        else:
            raise RuntimeError(f"Could not download {filename}") from last_error
        if first is None:
            first = path
    model = Path(first)
    if not model.is_file():
        raise FileNotFoundError(f"Sarvam-30B Q4_K_M shard is missing: {model}")
    return str(model)


def _stop_sarvam_server() -> None:
    process = _sarvam_server_state.get("process")
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _sarvam_server() -> str:
    process = _sarvam_server_state.get("process")
    if process is not None and process.poll() is None:
        return _sarvam_server_state["url"]

    bin_dir = _llama_cpp_dir()
    _SARVAM_CACHE.mkdir(parents=True, exist_ok=True)
    log_path = _SARVAM_CACHE / "llama-server.log"
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    command = [
        str(bin_dir / "llama-server.exe"),
        "--model",
        _sarvam_model_path(),
        "--fit",
        "on",
        "--flash-attn",
        "on",
        "--split-mode",
        "none",
        "--main-gpu",
        "0",
        "--device",
        "CUDA0",
        "--ctx-size",
        str(_SARVAM_CTX),
        "--parallel",
        "1",
        "--verbosity",
        "4",
        "--log-file",
        str(log_path),
        "--jinja",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--embeddings",
        "--pooling",
        "last",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if log_path.exists():
        log_path.unlink()
    error_path = log_path.with_suffix(".stderr")
    error_handle = error_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        command,
        cwd=bin_dir,
        stdout=error_handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    _sarvam_server_state["process"] = process
    _sarvam_server_state["url"] = url
    try:
        _wait_for_server(
            process,
            url,
            log_path,
            timeout=600,
            require_full_gpu=False,
        )
    except Exception:
        _stop_sarvam_server()
        _sarvam_server_state["process"] = None
        _sarvam_server_state["url"] = None
        raise
    match = _GPU_OFFLOAD_RE.search(_read_text(log_path))
    if match:
        print(f"Sarvam-30B offloaded {match.group(1)}/{match.group(2)} layers to the GPU")
    return url


def _sarvam_completion(url: str, prompt: str, temperature: float, external_CoT: bool) -> dict:
    # The template only adds <|nothink|>. This model still opens <think> and
    # spends the whole token budget there, which llama.cpp then hides.
    # Prefill a closed think block so generation starts at the answer.
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "<think></think>\n"},
    ]
    timeout = _completion_timeout(external_CoT)
    body = _llama_post(
        url,
        "/v1/chat/completions",
        {
            "model": "sarvam-30b",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": _max_tokens(external_CoT),
            "stop": ["<|end_of_turn|>"],
            "reasoning_budget_tokens": 0,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
        },
        timeout=timeout,
    )
    choice = body["choices"][0]
    record = _generation_record(url, messages[:1], choice, timeout=timeout)
    record["text"] = _THINK_PREFIX_RE.sub("", record["text"]).strip()
    return record


def ask_sarvam(rows: list[dict], native: bool, external_CoT: bool, temperature: float) -> list[dict]:
    get_prompt = _test_model_attr("get_prompt")
    language = _test_model_attr("LANGUAGE")

    url = _sarvam_server()
    outputs = []
    for row in rows:
        prompt = get_prompt(row, native, external_CoT)
        record = _sarvam_completion(url, prompt, temperature, external_CoT)
        response_text = " ".join(record["text"].splitlines())
        print(
            f"language={language} filetype={row['filetype']} native={native} "
            f"external_CoT={external_CoT} temperature={temperature} response={response_text}"
        )
        outputs.append(record)
    return outputs


atexit.register(_stop_sarvam_server)
