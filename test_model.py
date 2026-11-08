import argparse
import csv
import random
from pathlib import Path

import pandas as pd

from ask_navarasa import ask_navarasa
from ask_param import ask_param
from ask_sarvam import ask_sarvam

_LANGUAGES = ("Gujarati", "Marathi", "Punjabi", "Telugu", "Urdu")
_LANGUAGE_BY_NAME = {name.lower(): name for name in _LANGUAGES}
LANGUAGE = _LANGUAGES[0]
_NO_INFORMATION = "No Information Available"

_ZERO_SHOT_QA_PROMPT = """You are a multilingual factual QA assistant. Provide a short, precise answer (1-5 words). If the question is in English, answer in English. If the question is provided in a language other than English, interpret and answer it in that language.If no factual answer exists, OR you don’t know the answer, reply exactly: No Information Available
Question: {question}
Answer:"""

_ZERO_SHOT_QA_COT_PROMPT = """Your task is to answer the given question accurately in the language specified by the input.
Follow these steps:
Understand the meaning of the question.
Identify the key information needed to answer the question.
If the question is provided in a language other than English, interpret it in that language.
Reason through the question step by step, explaining the relevant facts and how they support the answer.
Verify that the proposed answer is factually supported and consistent with the question.
If no factual answer exists, or you do not know the answer, respond exactly: No Information Available.
Provide the final answer in the same language as the question.
Keep the final answer concise (1–5 words) and directly address the question.
Show your reasoning followed by the final answer using this format:
Reasoning: [Step-by-step explanation]
Final answer: [Concise answer]
Question: {question}"""

_MCQ_PROMPT = """You are a multilingual factual assistant. You will be given a multiple-choice question. If the question and options are provided in a language other than English, interpret it in that language and choose any one of the three options provided.

Each question is provided as follows:
- The question appears first
- Three answer choices are listed as A), B), and C)
- Your task is to select one of the three options. The response must contain only one letter: A, B, or C.
- Do not provide justifications.

Question: {question}
Option A: {Option A}
Option B: {Option B}
Option C: {Option C}

Answer:"""

_MCQ_COT_PROMPT = """You are a multilingual factual assistant. Interpret the question and all three options in the language provided. Select the most accurate option based on factual correctness and relevance.
Understand the question and identify the relevant facts.
Evaluate options A, B, and C independently. Accept or reject each option and briefly justify your judgment using factual evidence.
Compare the options and select the single best answer.
If uncertain, choose the best available option without inventing facts.
Keep the explanation concise and ensure the final selection matches your evaluation.
Use exactly this output format:
Reasoning:
Option A: [Accept or reject, with a brief explanation]
Option B: [Accept or reject, with a brief explanation]
Option C: [Accept or reject, with a brief explanation]
Final selection: [A, B, or C]
Answer: [A, B, or C]
Question: {question}
Option A: {Option A}
Option B: {Option B}
Option C: {Option C}"""


def _pick(row: pd.Series, name: str):
    if name not in row.index:
        return None
    value = row[name]
    if pd.isna(value):
        return None
    return value if isinstance(value, str) else str(value)


def _dicts_from_frame(frame: pd.DataFrame, file_type: str, native: bool) -> list[dict]:
    suffix = "_Native" if native else ""
    rows = []
    for _, row in frame.iterrows():
        if file_type == "MCQ":
            item = {
                f"Question{suffix}": _pick(row, "Question"),
                f"Option_A{suffix}": _pick(row, "Option A"),
                f"Option_B{suffix}": _pick(row, "Option B"),
                f"Option_C{suffix}": _pick(row, "Option C"),
                f"Correct_Answer{suffix}": _pick(row, "Correct Answer"),
            }
        elif file_type == "NonExi":
            item = {
                f"Question{suffix}": _pick(row, "Question"),
                f"Answer{suffix}": _NO_INFORMATION,
            }
        else:
            item = {
                f"Question{suffix}": _pick(row, "Question"),
                f"Answer{suffix}": _pick(row, "Answer"),
            }
        item["Domain"] = _pick(row, "Domain")
        item["filetype"] = file_type
        rows.append(item)
    return rows


def qa_prompt(row: dict, native: bool, external_CoT: bool) -> str:
    question = row["Question_Native"] if native else row["Question"]
    template = _ZERO_SHOT_QA_COT_PROMPT if external_CoT else _ZERO_SHOT_QA_PROMPT
    return template.format(question=question)


def mcq_prompt(row: dict, native: bool, external_CoT: bool) -> str:
    if native:
        question = row["Question_Native"]
        option_a = row["Option_A_Native"]
        option_b = row["Option_B_Native"]
        option_c = row["Option_C_Native"]
    else:
        question = row["Question"]
        option_a = row["Option_A"]
        option_b = row["Option_B"]
        option_c = row["Option_C"]
    template = _MCQ_COT_PROMPT if external_CoT else _MCQ_PROMPT
    return (
        template.replace("{question}", question)
        .replace("{Option A}", option_a)
        .replace("{Option B}", option_b)
        .replace("{Option C}", option_c)
    )


def get_prompt(row: dict, native: bool, external_CoT: bool) -> str:
    if row["filetype"] == "MCQ":
        return mcq_prompt(row, native, external_CoT)
    return qa_prompt(row, native, external_CoT)


def combine_sets(first: list[dict], second: list[dict]) -> list[dict]:
    combined = []
    for row in first + second:
        copied = dict(row)
        copied["filetype"] = "Both"
        combined.append(copied)
    random.Random(0).shuffle(combined)
    return combined


def _generated_fields(generated: dict) -> tuple[str, str, str, str]:
    return (
        generated.get("text", ""),
        generated.get("logits", ""),
        generated.get("token_probabilities", ""),
        generated.get("hidden_states", ""),
    )


def make_csv(
    rows: list[dict],
    temperature: float,
    native: bool,
    external_CoT: bool,
    generated_responses: list,
    temperature_folder: bool = False,
    *,
    output_file: str,
) -> Path:
    if not rows:
        raise ValueError("rows is empty")
    if len(generated_responses) != len(rows):
        raise ValueError(
            f"Expected {len(rows)} generated responses, got {len(generated_responses)}"
        )

    file_type = rows[0]["filetype"]
    suffix = "_Native" if native else ""
    if str(file_type).lower() == "mcq":
        headers = [
            "Question",
            "Option A",
            "Option B",
            "Option C",
            "Answer",
            "Generated Answer",
        ]
        field_keys = [
            f"Question{suffix}",
            f"Option_A{suffix}",
            f"Option_B{suffix}",
            f"Option_C{suffix}",
            f"Correct_Answer{suffix}",
        ]
    else:
        headers = ["Question", "Answer", "Generated Answer"]
        field_keys = [f"Question{suffix}", f"Answer{suffix}"]

    include_domain = any(
        row.get("Domain") is not None and str(row.get("Domain")).strip() != ""
        for row in rows
    )
    if include_domain:
        headers.insert(-1, "Domain")
        field_keys.append("Domain")
    headers.extend(["Logits", "Token Probabilities", "Hidden States"])

    project_root = _raw_dir().parent

    language_label = "Native" if native else "English"
    cot_label = "CoT" if external_CoT else "ZeroShot"
    output_dir = project_root / output_file / f"{language_label} ({cot_label})"
    if temperature_folder:
        output_dir = output_dir / f"Temperature ({format(temperature, 'g')})"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{cot_label}{file_type}[{LANGUAGE}].csv"

    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row, generated in zip(rows, generated_responses):
            values = []
            for key in field_keys:
                value = row.get(key)
                if value is None or str(value).strip() == "":
                    values.append("")
                else:
                    values.append(value)
            text, logits, probabilities, hidden_states = _generated_fields(generated)
            values.extend([text, logits, probabilities, hidden_states])
            writer.writerow(values)
    return output_path


def _raw_dir() -> Path:
    roots = []
    module_file = globals().get("__file__")
    if module_file:
        roots.append(Path(module_file).resolve().parent)
    roots.append(Path.cwd())
    for root in roots:
        candidate = root / "raw"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not find raw folder")


def _variant_path(raw_dir: Path, language: str, file_type: str, variant: str) -> Path | None:
    path = raw_dir / f"{language}_Indic{file_type} [{variant}].xlsx"
    return path if path.is_file() else None


def _merge_language_rows(english_rows: list[dict], native_rows: list[dict]) -> list[dict]:
    count = max(len(english_rows), len(native_rows))
    merged = []
    for index in range(count):
        row = {}
        if index < len(english_rows):
            row.update(english_rows[index])
        if index < len(native_rows):
            row.update(native_rows[index])
        merged.append(row)
    return merged


def _load_file_type(raw_dir: Path, language: str, file_type: str) -> list[dict]:
    english_path = _variant_path(raw_dir, language, file_type, "English")
    native_path = _variant_path(raw_dir, language, file_type, language)
    if english_path is None and native_path is None:
        print(f"skip {language} {file_type}: spreadsheet not found")
        return []

    english_frame = pd.read_excel(english_path) if english_path else None
    native_frame = pd.read_excel(native_path) if native_path else None
    if (
        english_frame is not None
        and native_frame is not None
        and len(english_frame) != len(native_frame)
    ):
        raise ValueError(
            f"{language} {file_type} English and {language} files have different lengths"
        )

    english_rows = []
    native_rows = []
    if english_frame is not None:
        english_rows = _dicts_from_frame(english_frame, file_type, False)
    else:
        print(f"skip {language} {file_type} English: spreadsheet not found")
    if native_frame is not None:
        native_rows = _dicts_from_frame(native_frame, file_type, True)
    else:
        print(f"skip {language} {file_type} {language}: spreadsheet not found")
    return _merge_language_rows(english_rows, native_rows)


def _side_ready(rows: list[dict], native: bool) -> bool:
    key = "Question_Native" if native else "Question"
    return any(row.get(key) not in (None, "") for row in rows)


def _load_eval_sets() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    raw_dir = _raw_dir()
    mcq = _load_file_type(raw_dir, LANGUAGE, "MCQ")
    nonexi = _load_file_type(raw_dir, LANGUAGE, "NonExi")
    wikiqa = _load_file_type(raw_dir, LANGUAGE, "WikiQA")
    both = combine_sets(nonexi, wikiqa) if nonexi and wikiqa else []
    return mcq, nonexi, wikiqa, both


def _run_model(
    ask,
    rows: list[dict],
    native: bool,
    external_CoT: bool,
    temperature: float,
    temperature_folder: bool = False,
    *,
    output_file: str,
) -> None:
    if not _side_ready(rows, native):
        return
    generated_answers = ask(rows, native, external_CoT, temperature)
    make_csv(
        rows,
        temperature,
        native,
        external_CoT,
        generated_answers,
        temperature_folder=temperature_folder,
        output_file=output_file,
    )


def _make_runner(ask, output_file: str):
    def run(
        rows: list[dict],
        native: bool,
        external_CoT: bool,
        temperature: float,
        temperature_folder: bool = False,
    ) -> None:
        _run_model(
            ask,
            rows,
            native,
            external_CoT,
            temperature,
            temperature_folder=temperature_folder,
            output_file=output_file,
        )

    return run


_MODEL_RUNNERS = {
    "param": _make_runner(ask_param, "Param 2 Outputs"),
    "sarvam": _make_runner(ask_sarvam, "Sarvam Outputs"),
    "navarasa": _make_runner(ask_navarasa, "Navarasa Outputs"),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Indic evals for one local model.")
    parser.add_argument(
        "model",
        type=str.lower,
        choices=tuple(_MODEL_RUNNERS),
        help="Model to run: param, sarvam, or navarasa.",
    )
    parser.add_argument(
        "language",
        type=str.lower,
        choices=(*_LANGUAGE_BY_NAME, "all"),
        help="Language to run, or all for Gujarati, Marathi, Punjabi, Telugu, and Urdu.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("english",),
        help="Pass english to run English zero-shot on MCQ, NonExi, and Both.",
    )
    return parser.parse_args(argv)


def _selected_languages(language: str) -> tuple[str, ...]:
    if language == "all":
        return _LANGUAGES
    return (_LANGUAGE_BY_NAME[language],)


def generate_outputs(run) -> None:
    # Edit the run() calls below to choose which CSVs are written.
    # Copy a call and change its arguments, or delete a call to skip that file.
    #
    # run(rows, native, external_CoT, temperature, temperature_folder=...)
    # rows: mcq, nonexi, wikiqa, or both. both is NonExi and WikiQA shuffled
    #   together. One call writes one file, such as ZeroShotMCQ[Gujarati].csv,
    #   under the model folder in Native (CoT), English (ZeroShot), and so on.
    # native: True uses the language questions. False uses the English questions.
    # external_CoT: True uses the chain-of-thought prompt and puts CoT in the
    #   path. False uses the zero-shot prompt and puts ZeroShot in the path.
    # temperature: value sent to the model. The folder path leaves it out.
    # temperature_folder: pass True to add a Temperature (value) subfolder.
    # A call is skipped when that set has no questions on the chosen side.
    mcq, nonexi, wikiqa, both = _load_eval_sets()
    temperature = 0

    # Native and English, zero-shot and chain-of-thought.
    for native in (True, False):
        for external_CoT in (True, False):
            run(nonexi, native, external_CoT, temperature)
            run(mcq, native, external_CoT, temperature)
    # Zero-shot only.
    for native in (True, False):
        run(wikiqa, native, False, temperature)
        run(both, native, False, temperature)


def main_english_zeroshot(run) -> None:
    mcq, nonexi, _wikiqa, both = _load_eval_sets()
    temperature = 0
    native = False
    external_CoT = False
    for rows in (mcq, nonexi, both):
        run(rows, native, external_CoT, temperature, temperature_folder=True)


if __name__ == "__main__":
    args = _parse_args()
    run = _MODEL_RUNNERS[args.model]
    for language in _selected_languages(args.language):
        LANGUAGE = language
        print(f"language={LANGUAGE}", flush=True)
        if args.mode == "english":
            main_english_zeroshot(run)
        else:
            generate_outputs(run)