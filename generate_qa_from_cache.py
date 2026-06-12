#!/usr/bin/env python3
"""Generate Q&A from already-fetched data/book_*.json files (no re-download)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from generate_qa_dataset import (
    FINAL_QA_FILE,
    MY_50_BOOK_IDS,
    OUTPUT_DIR,
    SYSTEM_PROMPT,
    clean_json_response,
    format_time,
)

QA_PROGRESS_FILE = Path("qa_progress.json")
DEFAULT_CPU_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_GPU_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_INPUT_CHARS = int(os.environ.get("QA_MAX_INPUT_CHARS", "2500"))


def load_qa_generator(model_name: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    has_cuda = torch.cuda.is_available()
    if model_name is None:
        model_name = DEFAULT_GPU_MODEL if has_cuda else DEFAULT_CPU_MODEL

    print(f"Loading {model_name} on {'CUDA' if has_cuda else 'CPU'}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs: dict = {"device_map": "auto"}
    pipe_kwargs: dict = {
        "max_new_tokens": int(os.environ.get("QA_MAX_NEW_TOKENS", "512")),
        "temperature": 0.2,
        "do_sample": True,
    }

    if has_cuda:
        try:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        except ImportError:
            load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **pipe_kwargs)

    def generate_qa_offline(text_content: str) -> str:
        truncated = text_content[:MAX_INPUT_CHARS]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f'نص الصفحة:\n"""\n{truncated}\n"""\n\n'
                    "يرجى توليد الأسئلة والأجوبة بصيغة JSON."
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt)
        return outputs[0]["generated_text"][len(prompt) :].strip()

    return generate_qa_offline


def page_key(book_id: int, page_id: int | None, page_number: int | None) -> str:
    return f"{book_id}:{page_id or page_number}"


def load_qa_progress() -> dict:
    if QA_PROGRESS_FILE.is_file():
        return json.loads(QA_PROGRESS_FILE.read_text(encoding="utf-8"))
    processed: set[str] = set()
    all_questions: list = []
    if FINAL_QA_FILE.is_file():
        try:
            all_questions = json.loads(FINAL_QA_FILE.read_text(encoding="utf-8"))
            for item in all_questions:
                ref = item.get("reference", {})
                processed.add(
                    page_key(ref.get("book_id"), ref.get("page_id"), ref.get("page_number"))
                )
        except json.JSONDecodeError:
            pass
    return {"processed_pages": sorted(processed), "all_questions": all_questions}


def save_qa_progress(processed_pages: set[str], all_questions: list) -> None:
    QA_PROGRESS_FILE.write_text(
        json.dumps(
            {"processed_pages": sorted(processed_pages), "all_questions": all_questions},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    FINAL_QA_FILE.write_text(json.dumps(all_questions, ensure_ascii=False, indent=2), encoding="utf-8")


def run_qa_from_cache(
    book_ids: list[int],
    generate_qa_offline,
    resume: bool = True,
    limit_pages: int | None = None,
) -> None:
    progress = load_qa_progress() if resume else {"processed_pages": [], "all_questions": []}
    processed_pages = set(progress.get("processed_pages", []))
    all_questions: list = list(progress.get("all_questions", []))

    total_pages = 0
    for book_id in book_ids:
        book_path = OUTPUT_DIR / f"book_{book_id}.json"
        if book_path.is_file():
            book_data = json.loads(book_path.read_text(encoding="utf-8"))
            total_pages += len(book_data.get("pages", []))

    pages_done = len(processed_pages)
    start_time = time.time()
    generated_this_run = 0

    print(f"\nGenerating Q&A for {total_pages} pages ({pages_done} already done)...")

    for book_idx, book_id in enumerate(book_ids, 1):
        book_path = OUTPUT_DIR / f"book_{book_id}.json"
        if not book_path.is_file():
            print(f"Missing data/book_{book_id}.json — run fetch first.")
            continue

        book_data = json.loads(book_path.read_text(encoding="utf-8"))
        title = book_data.get("title", f"Book {book_id}")
        pages = book_data.get("pages", [])
        if limit_pages:
            pages = pages[:limit_pages]

        print(f"\nBook [{book_idx}/{len(book_ids)}] {book_id}: {title} ({len(pages)} pages)")

        for page_idx, page in enumerate(pages, 1):
            pid = page.get("page_id") or page.get("id")
            page_num = page.get("page_number") or page.get("page")
            key = page_key(book_id, pid, page_num)

            if key in processed_pages:
                continue

            text = page.get("text", "")
            if not text or len(text) < 200:
                processed_pages.add(key)
                continue

            page_start = time.time()
            try:
                raw_response = generate_qa_offline(text)
                qa_pairs = clean_json_response(raw_response)
            except Exception as exc:
                print(f"  Error on page {key}: {exc}")
                qa_pairs = []

            for pair in qa_pairs:
                if not isinstance(pair, dict):
                    continue
                pair["reference"] = {
                    "book_id": int(book_id),
                    "book_title": title,
                    "page_id": pid,
                    "page_number": page_num,
                    "part_name": page.get("part_name", "1"),
                    "breadcrumbs": page.get("breadcrumbs", []),
                }
                all_questions.append(pair)

            processed_pages.add(key)
            generated_this_run += 1
            pages_done += 1
            page_duration = time.time() - page_start
            elapsed = time.time() - start_time
            avg = elapsed / max(generated_this_run, 1)
            remaining = (total_pages - pages_done) * avg

            print(
                f"  Page [{page_idx}/{len(pages)}] {key} | "
                f"+{len(qa_pairs)} Q&As | {page_duration:.1f}s | "
                f"Total: {len(all_questions)} | "
                f"Done {pages_done}/{total_pages} | ETA {format_time(remaining)}"
            )

            save_qa_progress(processed_pages, all_questions)

    elapsed = time.time() - start_time
    print(f"\nQ&A generation complete: {len(all_questions)} pairs in {format_time(elapsed)}")
    print(f"Saved to {FINAL_QA_FILE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Q&A from cached book JSON files.")
    parser.add_argument("--book-ids", nargs="+", type=int, default=MY_50_BOOK_IDS)
    parser.add_argument("--model", default=os.environ.get("QA_MODEL"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit-pages-per-book", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_qa = load_qa_generator(args.model)
    run_qa_from_cache(
        book_ids=args.book_ids,
        generate_qa_offline=generate_qa,
        resume=not args.no_resume,
        limit_pages=args.limit_pages_per_book,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
