import atexit
import re
import subprocess
from pathlib import Path

from llama_server import (
    _GPU_OFFLOAD_RE,
    _completion_timeout,
    _final_hidden_state,
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

_PARAM_REPO = "arunmcops/Param2-17B-A2.4B-Thinking-GGUF"
_PARAM_FILE = "Param2-17B-A2.4B-Thinking-Q4_K_M.gguf"
_PARAM_CTX = 8192
_PARAM_CACHE = Path.home() / ".cache" / "param2"
_PARAM_THINK_BLOCK_RE = re.compile(
    r"<\s*\|\s*think\s*\|\s*>.*?<\s*/\s*\|\s*think\s*\|\s*>|<\s*think\s*>.*?<\s*/\s*think\s*>",
    re.IGNORECASE | re.DOTALL,
)
_PARAM_THINKING_RE = re.compile(
    r"\A\s*<think>\s*# Wait a sec\.\. Thinking hard\.\.\.\s*",
    re.IGNORECASE,
)
_PARAM_THINK_TAG_RE = re.compile(
    r"<\s*/?\s*(?:\|\s*)?think(?:\s*\|)?\s*>",
    re.IGNORECASE,
)
_param_server_state: dict = {"process": None, "url": None}


def _param_answer(content: str) -> str:
    outside = _PARAM_THINK_BLOCK_RE.sub("", content)
    outside = _PARAM_THINKING_RE.sub("", outside, count=1)
    outside = re.sub(
        r"\A\s*(?:<\s*\|\s*think\s*\|\s*>|<\s*think\s*>)\s*",
        "",
        outside,
        count=1,
        flags=re.IGNORECASE,
    )
    outside = re.sub(r"\A\s*>\s*", "", outside).strip()
    if outside:
        return outside
    # The whole reply was inside <think>, so removing the block left nothing.
    return _PARAM_THINK_TAG_RE.sub("", content).strip()


def _param_model_path() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=_PARAM_REPO, filename=_PARAM_FILE)


def _stop_param_server() -> None:
    process = _param_server_state.get("process")
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _param_server() -> str:
    process = _param_server_state.get("process")
    if process is not None and process.poll() is None:
        return _param_server_state["url"]

    bin_dir = _llama_cpp_dir()
    _PARAM_CACHE.mkdir(parents=True, exist_ok=True)
    log_path = _PARAM_CACHE / "llama-server.log"
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    command = [
        str(bin_dir / "llama-server.exe"),
        "--model",
        _param_model_path(),
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
        str(_PARAM_CTX),
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
        "--no-reasoning-preserve",
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
    _param_server_state["process"] = process
    _param_server_state["url"] = url
    try:
        _wait_for_server(
            process,
            url,
            log_path,
            timeout=300,
            require_full_gpu=False,
        )
    except Exception:
        _stop_param_server()
        _param_server_state["process"] = None
        _param_server_state["url"] = None
        raise
    match = _GPU_OFFLOAD_RE.search(_read_text(log_path))
    if match:
        print(f"Param2 offloaded {match.group(1)}/{match.group(2)} layers to the GPU")
    return url


def _param_visible_answer(choice: dict) -> str:
    message = choice.get("message") or {}
    for key in ("content", "reasoning_content", "reasoning"):
        text = _param_answer(message.get(key) or "")
        if text:
            return text
    tokens = []
    for item in (choice.get("logprobs") or {}).get("content") or []:
        if isinstance(item, dict) and item.get("token"):
            tokens.append(item["token"])
    return _param_answer("".join(tokens))


def _param_completion(url: str, prompt: str, temperature: float, external_CoT: bool) -> dict:
    messages = [{"role": "user", "content": prompt}]
    timeout = _completion_timeout(external_CoT)
    body = _llama_post(
        url,
        "/v1/chat/completions",
        {
            "model": "param2",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": _max_tokens(external_CoT),
            "reasoning_budget_tokens": 0,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
        },
        timeout=timeout,
    )
    choice = body["choices"][0]
    record = _generation_record(url, messages, choice, timeout=timeout)
    visible = _param_visible_answer(choice)
    if visible and not (choice.get("message") or {}).get("content"):
        record["hidden_states"] = _final_hidden_state(url, messages, visible, timeout)
    record["text"] = visible
    return record


def ask_param(rows: list[dict], native: bool, external_CoT: bool, temperature: float) -> list[dict]:
    get_prompt = _test_model_attr("get_prompt")
    language = _test_model_attr("LANGUAGE")

    url = _param_server()
    outputs = []
    for row in rows:
        prompt = get_prompt(row, native, external_CoT)
        record = _param_completion(url, prompt, temperature, external_CoT)
        response_text = " ".join(record["text"].splitlines())
        print(
            f"language={language} filetype={row['filetype']} native={native} "
            f"external_CoT={external_CoT} temperature={temperature} response={response_text}"
        )
        outputs.append(record)
    return outputs


atexit.register(_stop_param_server)
