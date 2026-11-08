# Model testing

This folder runs factual questions through a local model and writes one CSV per question set. Each row keeps the original question, the reference answer, the model’s reply, per-token scores, and one hidden-state vector for the finished reply.

Use Python 3.10 or newer. From this folder:

```
python -m pip install -r requirements.txt
```

Put the IndicWikiQA, NonExi, and MCQ spreadsheets for the desired language in the `raw` folder before running a test on these datasets.

The included ask files run on Windows with an NVIDIA GPU as CUDA device 0 and a driver that can run CUDA 12.4. The first run downloads a llama.cpp build and the model weights into your home directory. Sarvam is a 30B model in six files, Param is 17B, and Navarasa is 7B. Navarasa must fit entirely on the GPU. One language command reads every row of the sheets it runs, and a chain-of-thought reply can be up to 4096 tokens. Details for other systems and GPUs are in [Operating system and GPU](#operating-system-and-gpu).

## Files

| File | Role |
|---|---|
| `test_model.py` | Loads the spreadsheets, builds prompts, calls one model, and writes CSVs. This is the file you run and the file you edit to choose which sets are generated. |
| `llama_server.py` | Shared helper for a local llama.cpp server: download the CUDA build, start `llama-server`, send a completion, and turn the reply into the CSV record. |
| `ask_param.py`, `ask_sarvam.py`, `ask_navarasa.py` | One file per model. Each file is an example of the same ask-function shape, described below. |
| `raw/` | Input spreadsheets. |

## Running a test

From this folder:

```
python test_model.py navarasa gujarati
python test_model.py param all
python test_model.py sarvam urdu english
```

The included ask files download a GGUF from Hugging Face and start llama.cpp on the local GPU. That path needs `pandas`, `openpyxl`, and `huggingface_hub`, plus a CUDA GPU. A new ask file can call a different backend and only needs whatever that backend requires.

## Operating system and GPU

No path in this folder names a particular computer or user. Downloads and logs go under that user's home directory (`Path.home() / ".cache"`), and each server listens on `127.0.0.1` with a free port.

The included server setup is written for Windows and the first NVIDIA GPU:

- `llama_server.py` downloads the Windows CUDA 12.4 archives for llama.cpp release `b11982` (`llama-b11982-bin-win-cuda-12.4-x64.zip` and `cudart-llama-bin-win-cuda-12.4-x64.zip`). `_llama_cpp_dir` then looks for `llama-server.exe` and a `cudart64_*.dll`.
- Each ask file starts `llama-server.exe` with `--main-gpu 0` and `--device CUDA0`. `ask_navarasa.py` also pins `token_embd.weight` to `CUDA0`.
- `CREATE_NO_WINDOW` is only applied when Python is running on Windows. Other systems pass no creation flag.

To run on another operating system, edit `llama_server.py` and the server command in each ask file you use:

1. On the [b11982 release](https://github.com/ggml-org/llama.cpp/releases/tag/b11982), choose the archive for that OS and GPU backend. Put those file names in `_LLAMA_CPP_BIN_ZIP` and `_LLAMA_CPP_CUDART_ZIP`. A build that ships its libraries inside one archive only needs that one download. Drop the second name from the loop in `_llama_cpp_dir`.
2. Point `_llama_cpp_dir` at `llama-server` instead of `llama-server.exe`. On Linux and macOS, remove the `cudart64_*.dll` check. That check is what decides the Windows CUDA build is complete.
3. In the ask file, start `llama-server` from that same directory instead of `llama-server.exe`.

To use a different GPU, change the device flags in that ask file's server command. They have to name the same device:

| Goal | Change |
|---|---|
| Second NVIDIA GPU | `--main-gpu 1`, `--device CUDA1`, and in `ask_navarasa.py` `token_embd.weight=CUDA1` |
| Another backend (Metal, Vulkan, ROCm) | Use that backend's archive, then set `--device` to the name printed in the server log |
| CPU | Use a CPU archive, remove `--device`, `--main-gpu`, and Navarasa's `--n-gpu-layers all` and `--override-tensor` lines |

`ask_navarasa.py` calls `_wait_for_server` with the default `require_full_gpu=True`, so Navarasa stops unless every layer is on the GPU. `ask_param.py` and `ask_sarvam.py` pass `require_full_gpu=False`. A CPU run of Navarasa needs `require_full_gpu=False` as well.

### Spreadsheets

Put workbooks in `raw/`. A name has three parts:

```
{Language}_Indic{MCQ|NonExi|WikiQA} [{English|Language}].xlsx
```

Examples: `Gujarati_IndicMCQ [English].xlsx` and `Gujarati_IndicMCQ [Gujarati].xlsx`.

| Set | Columns that are read |
|---|---|
| MCQ | `Question`, `Option A`, `Option B`, `Option C`, `Correct Answer`, `Domain` |
| WikiQA | `Question`, `Answer`, `Domain` |
| NonExi | `Question`, `Domain`. The reference answer is always `No Information Available`. |

English and native workbooks for the same language and set are paired by row number. A missing workbook is skipped, and that side of the run is skipped with it. If both exist and the row counts differ, the run stops.

The languages the command line accepts are Gujarati, Marathi, Punjabi, Telugu, and Urdu. Add a name to `_LANGUAGES` at the top of `test_model.py` to accept another one, and add matching files under `raw/`.

Prompt text is the four constants at the top of `test_model.py`. MCQ rows use the MCQ prompts. NonExi, WikiQA, and the combined set use the QA prompts.

## Command line

```
python test_model.py <model> <language> [english]
```

`model` chooses which ask function runs. The current names are `param`, `sarvam`, and `navarasa`. Matching is case-insensitive. Each name also selects the output folder registered next to that function in `_MODEL_RUNNERS`.

`language` is one language, or `all`. `all` runs Gujarati, Marathi, Punjabi, Telugu, and Urdu in that order. The active language is stored in `LANGUAGE` and is used in the CSV file name.

`english` is optional. Without it, the script calls `generate_outputs`. With it, the script runs English zero-shot only, on MCQ, NonExi, and the combined NonExi+WikiQA set, and those CSVs go in a `Temperature (0)` folder.

## Changing `generate_outputs`

`generate_outputs` in `test_model.py` is the list of CSVs for a normal run. Each `run(...)` call writes one file. Copy a call to add a file, delete a call to skip one, or change the arguments.

```
run(rows, native, external_CoT, temperature, temperature_folder=False)
```

| Argument | What it changes |
|---|---|
| `rows` | `mcq`, `nonexi`, `wikiqa`, or `both`. `both` is NonExi and WikiQA copied together, labeled `Both`, and shuffled with seed 0, so the row order is the same on every machine. |
| `native` | `True` uses the language questions and writes under `Native (...)`. `False` uses the English questions and writes under `English (...)`. |
| `external_CoT` | `True` uses the chain-of-thought prompt and puts `CoT` in the path. `False` uses the zero-shot prompt and puts `ZeroShot` in the path. |
| `temperature` | Value passed through to the ask function. |
| `temperature_folder` | Leave it off to keep the CSV in the language and prompt folder. Pass `True` to add a `Temperature (value)` subfolder. |

A call is skipped when that set has no questions on the chosen side.

The current function writes NonExi and MCQ in both languages, with and without chain-of-thought, then WikiQA and `both` in both languages as zero-shot only. Temperature is `0`, and the temperature folder is left off.

Files land under the model folder chosen by `_MODEL_RUNNERS`:

```
{Model Outputs}/{Native|English} ({CoT|ZeroShot})/{CoT|ZeroShot}{Set}[{Language}].csv
```

For example, `Navarasa Outputs/English (ZeroShot)/ZeroShotMCQ[Gujarati].csv`.

An MCQ file has `Question`, `Option A`, `Option B`, `Option C`, `Answer`, then `Domain` when any row has one, then `Generated Answer`, `Logits`, `Token Probabilities`, and `Hidden States`. Other sets have `Question`, `Answer`, optional `Domain`, and the same four generated columns.

## Adding an ask file

An ask file wraps one model. `ask_param.py`, `ask_sarvam.py`, and `ask_navarasa.py` are working examples: each one exposes a single function the runner can call. A new model is a new file with that same function, plus two lines in `test_model.py`.

The function signature is:

```python
def ask_example(rows: list[dict], native: bool, external_CoT: bool, temperature: float) -> list[dict]:
    ...
```

`rows` is the question set for this call. Read prompts and the active language from `test_model` at call time, because `python test_model.py` loads that file as `__main__`:

```python
from llama_server import _test_model_attr

def ask_example(rows, native, external_CoT, temperature):
    get_prompt = _test_model_attr("get_prompt")
    language = _test_model_attr("LANGUAGE")
    outputs = []
    for row in rows:
        prompt = get_prompt(row, native, external_CoT)
        outputs.append(complete(prompt, temperature))
    return outputs
```

`get_prompt(row, native, external_CoT)` returns the MCQ or QA prompt for that row. Return one record per input row, in the same order. Each record is a dict:

| Key | Value |
|---|---|
| `text` | The reply written to `Generated Answer`. |
| `logits` | JSON text. One object per generated token, with `token`, `logprob`, and `top_logprobs` for the top five alternatives. |
| `token_probabilities` | JSON text with the same tokens. Each `logprob` is also stored as `probability` (`exp` of the log probability). |
| `hidden_states` | JSON text of one vector: the hidden state at the last token after the prompt and the full reply. |

A missing key is written as an empty cell, so a backend that cannot return scores can still fill `text`. The list length must match `rows`.

The included ask files use `llama_server.py` for this record. `_generation_record` builds all four fields from a llama.cpp chat completion that requested `logprobs` and `top_logprobs` (`TOP_LOGPROBS` is 5). A different API can build the same dict itself.

Register the function in `test_model.py`:

```python
from ask_example import ask_example

_MODEL_RUNNERS = {
    "param": _make_runner(ask_param, "Param 2 Outputs"),
    "sarvam": _make_runner(ask_sarvam, "Sarvam Outputs"),
    "navarasa": _make_runner(ask_navarasa, "Navarasa Outputs"),
    "example": _make_runner(ask_example, "Example Outputs"),
}
```

The new command is `python test_model.py example gujarati`. CSVs go in `Example Outputs`.
