import atexit
import subprocess
from pathlib import Path

from llama_server import (
    _completion_timeout,
    _free_port,
    _generation_record,
    _llama_cpp_dir,
    _llama_post,
    _max_tokens,
    _test_model_attr,
    _wait_for_server,
    TOP_LOGPROBS,
)

_NAVARASA_REPO = "Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa-2.0-gguf"
_NAVARASA_FILE = "Q4_K_M.gguf"
_NAVARASA_CTX = 4608
_NAVARASA_CACHE = Path.home() / ".cache" / "navarasa"
_NAVARASA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "### Instruction:\n{{ message['content'] }}\n\n### Input:\n\n### Response:\n"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] }}"
    "{% endif %}"
    "{% endfor %}"
)
_navarasa_server_state: dict = {"process": None, "url": None}


def _navarasa_model_path() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=_NAVARASA_REPO, filename=_NAVARASA_FILE)


def _stop_navarasa_server() -> None:
    process = _navarasa_server_state.get("process")
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _navarasa_server() -> str:
    process = _navarasa_server_state.get("process")
    if process is not None and process.poll() is None:
        return _navarasa_server_state["url"]

    bin_dir = _llama_cpp_dir()
    template_path = _NAVARASA_CACHE / "navarasa-chat-template.jinja"
    template_path.write_text(_NAVARASA_CHAT_TEMPLATE, encoding="utf-8")
    log_path = _NAVARASA_CACHE / "llama-server.log"
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    command = [
        str(bin_dir / "llama-server.exe"),
        "--model",
        _navarasa_model_path(),
        "--n-gpu-layers",
        "all",
        "--flash-attn",
        "on",
        "--fit",
        "off",
        "--split-mode",
        "none",
        "--main-gpu",
        "0",
        "--device",
        "CUDA0",
        "--override-tensor",
        "token_embd.weight=CUDA0",
        "--ctx-size",
        str(_NAVARASA_CTX),
        "--parallel",
        "1",
        "--verbosity",
        "4",
        "--log-file",
        str(log_path),
        "--jinja",
        "--chat-template-file",
        str(template_path),
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
    _navarasa_server_state["process"] = process
    _navarasa_server_state["url"] = url
    try:
        _wait_for_server(process, url, log_path)
    except Exception:
        _stop_navarasa_server()
        _navarasa_server_state["process"] = None
        _navarasa_server_state["url"] = None
        raise
    return url


def _navarasa_completion(url: str, prompt: str, temperature: float, external_CoT: bool) -> dict:
    messages = [{"role": "user", "content": prompt}]
    timeout = _completion_timeout(external_CoT)
    body = _llama_post(
        url,
        "/v1/chat/completions",
        {
            "model": "navarasa",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": _max_tokens(external_CoT),
            "stop": ["### Instruction:"],
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
        },
        timeout=timeout,
    )
    return _generation_record(url, messages, body["choices"][0], timeout=timeout)


def ask_navarasa(rows: list[dict], native: bool, external_CoT: bool, temperature: float) -> list[dict]:
    get_prompt = _test_model_attr("get_prompt")
    language = _test_model_attr("LANGUAGE")

    url = _navarasa_server()
    outputs = []
    for row in rows:
        prompt = get_prompt(row, native, external_CoT)
        record = _navarasa_completion(url, prompt, temperature, external_CoT)
        response_text = " ".join(record["text"].splitlines())
        print(
            f"language={language} filetype={row['filetype']} native={native} "
            f"external_CoT={external_CoT} temperature={temperature} response={response_text}"
        )
        outputs.append(record)
    return outputs


atexit.register(_stop_navarasa_server)
