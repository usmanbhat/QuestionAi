#!/usr/bin/env python3
"""
Fetch 50 ketabonline.com books, parse each page, and generate Q&A pairs
for AI model training. Uses proxy rotation to avoid 403 blocks.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

from proxy_client import ProxySession, load_proxy_list

OUTPUT_DIR = Path("./data")
FINAL_QA_FILE = Path("generated_questions_dataset.json")
PROGRESS_FILE = Path("pipeline_progress.json")

MY_50_BOOK_IDS = [
    5171, 10653, 6267, 5172, 1801, 157, 19098, 2700, 5181, 54481,
    6455, 14380, 20590, 2115, 5901, 4850, 915, 1098, 2689, 917,
    2532, 406, 5626, 6060, 2817, 1237, 5168, 1109, 5585, 2913,
    2839, 5174, 6051, 97113, 5169, 2213, 5186, 4709, 4018, 1130,
    2486, 5166, 90607, 16740, 107508, 25743, 2776, 41331, 10724, 14201,
]

SYSTEM_PROMPT = """أنت خبير في أصول الفقه الإسلامي ومختص في توليد بيانات تدريب نماذج الذكاء الاصطناعي (Synthetic Data Generation).
مهمتك هي قراءة نص الصفحة المقدمة من كتاب أصول الفقه، وتوليد أسئلة وإجابات دقيقة ومفصلة مستخلصة تماماً من النص.

لكل سؤال تقوم بتوليده، يجب تحديد ما يلي:
1. السؤال (question): بصيغة عربية فصحى دقيقة وواضحة.
2. الإجابة (answer): إجابة مفصلة وصحيحة ومباشرة من النص المرفق.
3. نوع السؤال (reason): تصنيف نوع السؤال/الهدف منه (مثال: definition, comparison, ruling, evidence, conditions, pillars, components, legal_reasoning, hadith_analysis, application).

الشروط:
- قم بتوليد ما بين 1 إلى 3 أسئلة لكل صفحة.
- يجب أن تكون الإجابة مستمدة بالكامل من النص المرفق، ولا تبتكر معلومات غير موجودة فيه.
- يجب إرجاع المخرجات بصيغة JSON فقط كقائمة من الكائنات (Array of Objects) بالصيغة المحددة أدناه، دون أي نصوص تمهيدية أو تعليقات خارجية."""


def clean_html_to_markdown(html_content: str) -> str:
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    lines: list[str] = []
    for child in soup.children:
        if child.name is None:
            text = child.strip()
            if text:
                lines.append(text)
            continue
        if child.name == "div" and "g-title" in child.get("class", []):
            title_text = child.get_text().strip()
            lines.append(f"\n## {title_text}\n")
        elif child.name == "p":
            title_div = child.find("div", class_="g-title")
            if title_div:
                title_text = title_div.get_text().strip()
                lines.append(f"\n## {title_text}\n")
                title_div.decompose()
            p_text = ""
            for sub in child.children:
                if sub.name is None:
                    p_text += str(sub)
                elif sub.name == "span" and "g-list" in sub.get("class", []):
                    p_text += f"**{sub.get_text().strip()}** "
                elif sub.name == "span" and "g-square-brackets" in sub.get("class", []):
                    p_text += sub.get_text()
                elif sub.name == "span" and "g-quotes" in sub.get("class", []):
                    p_text += f'"{sub.get_text().strip()}"'
                elif sub.name == "a":
                    pass
                else:
                    p_text += sub.get_text()
            p_text = re.sub(r"[ \t]+", " ", p_text.strip())
            if p_text:
                lines.append(p_text)
        else:
            text = child.get_text().strip()
            if text:
                lines.append(text)
    return "\n".join(lines).strip()


def build_all_page_breadcrumbs(index_data: list, total_pages: int) -> dict[int, list]:
    start_crumbs: dict[int, list[list[str]]] = {}

    def traverse(nodes: list, current_crumbs: list[str]) -> None:
        for node in nodes:
            page_id = node.get("page_id")
            title = node.get("title")
            node_crumbs = current_crumbs + [title] if title else current_crumbs
            if page_id:
                start_crumbs.setdefault(page_id, []).append(node_crumbs)
            children = node.get("children", [])
            if children:
                traverse(children, node_crumbs)

    traverse(index_data, [])
    for page_id in start_crumbs:
        start_crumbs[page_id] = sorted(start_crumbs[page_id], key=len, reverse=True)[:1]

    page_to_crumbs: dict[int, list] = {}
    current: list = []
    for pid in range(1, total_pages + 1):
        if pid in start_crumbs:
            current = start_crumbs[pid][0]
        page_to_crumbs[pid] = current
    return page_to_crumbs


def fetch_book_metadata(session: ProxySession, book_id: int) -> dict:
    url = f"https://backend.ketabonline.com/api/v2/books/{book_id}"
    response = session.get(url)
    return response.json().get("data", {})


def download_zip_data(session: ProxySession, zip_url: str) -> dict:
    response = session.get(zip_url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        json_filenames = [name for name in archive.namelist() if name.endswith(".json")]
        if not json_filenames:
            raise ValueError("No JSON file found in archive")
        with archive.open(json_filenames[0]) as handle:
            return json.loads(handle.read().decode("utf-8"))


def clean_json_response(raw_text: str) -> list:
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    try:
        parsed = json.loads(cleaned.strip())
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def format_time(seconds: float) -> str:
    if seconds <= 0:
        return "Calculating..."
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts: list[str] = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def load_progress() -> dict:
    if PROGRESS_FILE.is_file():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"completed_books": [], "all_questions": []}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def load_qa_generator():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    model_name = os.environ.get("QA_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    print(f"[Step 1/2] Loading {model_name}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.2,
        do_sample=True,
    )

    def generate_qa_offline(text_content: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f'نص الصفحة:\n"""\n{text_content}\n"""\n\n'
                    "يرجى توليد الأسئلة والأجوبة بصيغة JSON."
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt)
        return outputs[0]["generated_text"][len(prompt) :].strip()

    return generate_qa_offline


def run_pipeline(
    book_ids: list[int],
    session: ProxySession,
    generate_qa_offline=None,
    min_page_chars: int = 200,
    limit_pages_per_book: int | None = None,
    resume: bool = True,
) -> None:
    print("\n[Step 2/2] Starting processing pipeline...")
    if generate_qa_offline is None:
        print("  Mode: fetch + parse only (no Q&A generation). Use --generate-qa on a GPU machine.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    progress = load_progress() if resume else {"completed_books": [], "all_questions": []}
    completed_books = set(progress.get("completed_books", []))
    all_questions: list = list(progress.get("all_questions", []))

    total_books = len(book_ids)
    pages_processed_total = 0
    pipeline_start_time = time.time()

    for idx, book_id in enumerate(book_ids, 1):
        if book_id in completed_books:
            print(f"\nSkipping book {book_id} (already completed).")
            continue

        print("\n=======================================================")
        print(f"Book [{idx}/{total_books}] - Downloading Book ID: {book_id}...")
        print("=======================================================")

        try:
            metadata = fetch_book_metadata(session, book_id)
            title = metadata.get("title", f"Book {book_id}")
            print(f"Title: {title}")

            zip_url = metadata.get("files", {}).get("data", {}).get("url")
            if not zip_url:
                print(f"Skipping book {book_id}: No direct download zip found.")
                continue

            zip_json = download_zip_data(session, zip_url)
            pages = zip_json.get("pages", [])
            breadcrumbs_map = build_all_page_breadcrumbs(zip_json.get("index", []), len(pages))

            pages_to_run = []
            for page in pages:
                cleaned_text = clean_html_to_markdown(page.get("content", ""))
                if cleaned_text and len(cleaned_text) >= min_page_chars:
                    pages_to_run.append(page)

            if limit_pages_per_book:
                pages_to_run = pages_to_run[:limit_pages_per_book]

            num_pages_to_run = len(pages_to_run)
            print(f"Successfully loaded book. Processing {num_pages_to_run} pages...")

            book_output = {
                "book_id": book_id,
                "title": title,
                "pages": [],
            }

            for page_idx, page in enumerate(pages_to_run, 1):
                page_start_time = time.time()

                page_id = page.get("id")
                page_num = page.get("page")
                cleaned_text = clean_html_to_markdown(page.get("content", ""))
                part_info = page.get("part", {})
                part_name = part_info.get("name", "1") if isinstance(part_info, dict) else str(part_info)
                crumbs = breadcrumbs_map.get(page_id, [])

                page_record = {
                    "page_id": page_id,
                    "page_number": page_num,
                    "part_name": part_name,
                    "breadcrumbs": crumbs,
                    "text": cleaned_text,
                    "qa_pairs": [],
                }

                if generate_qa_offline is not None:
                    raw_response = generate_qa_offline(cleaned_text)
                    qa_pairs = clean_json_response(raw_response)
                    for pair in qa_pairs:
                        pair["reference"] = {
                            "book_id": int(book_id),
                            "book_title": title,
                            "page_number": page_num,
                            "part_name": part_name,
                            "breadcrumbs": crumbs,
                        }
                        all_questions.append(pair)
                        page_record["qa_pairs"].append(pair)

                book_output["pages"].append(page_record)
                pages_processed_total += 1
                page_duration = time.time() - page_start_time
                total_elapsed = time.time() - pipeline_start_time

                avg_time_per_page = total_elapsed / max(pages_processed_total, 1)
                remaining_pages_in_current_book = num_pages_to_run - page_idx
                remaining_books = total_books - idx
                estimated_remaining_pages_total = remaining_pages_in_current_book + (remaining_books * 300)
                eta_seconds = estimated_remaining_pages_total * avg_time_per_page

                print(
                    f"  -> Page [{page_idx}/{num_pages_to_run}] (Book Page: {page_num}) "
                    f"processed in {page_duration:.1f}s."
                )
                print(
                    f"     [Book {idx}/{total_books} | Total Generated: {len(all_questions)} Q&As | "
                    f"Elapsed: {format_time(total_elapsed)} | Est. Remaining: {format_time(eta_seconds)}]"
                )

                with open(FINAL_QA_FILE, "w", encoding="utf-8") as handle:
                    json.dump(all_questions, handle, ensure_ascii=False, indent=2)

            book_path = OUTPUT_DIR / f"book_{book_id}.json"
            book_path.write_text(json.dumps(book_output, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  Saved parsed book to {book_path}")

            completed_books.add(book_id)
            progress["completed_books"] = sorted(completed_books)
            progress["all_questions"] = all_questions
            save_progress(progress)

        except Exception as exc:
            print(f"Error processing book {book_id}: {exc}")

    write_manifest(book_ids)
    total_elapsed = time.time() - pipeline_start_time
    print(f"\nPipeline Complete! Generated {len(all_questions)} questions saved in: {FINAL_QA_FILE}")
    print(f"Parsed books saved under: {OUTPUT_DIR}")
    print(f"Total time elapsed: {format_time(total_elapsed)}")


def write_manifest(book_ids: list[int]) -> None:
    manifest = {"books": [], "total_pages": 0, "total_books": 0}
    for book_id in book_ids:
        book_path = OUTPUT_DIR / f"book_{book_id}.json"
        if not book_path.is_file():
            continue
        book_data = json.loads(book_path.read_text(encoding="utf-8"))
        page_count = len(book_data.get("pages", []))
        manifest["books"].append(
            {
                "book_id": book_id,
                "title": book_data.get("title"),
                "page_count": page_count,
                "file": str(book_path),
            }
        )
        manifest["total_pages"] += page_count
    manifest["total_books"] = len(manifest["books"])
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest['total_books']} books, {manifest['total_pages']} pages -> {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ketabonline books and generate Q&A training data.")
    parser.add_argument(
        "--book-ids",
        nargs="+",
        type=int,
        default=MY_50_BOOK_IDS,
        help="Book IDs to process (default: all 50 books).",
    )
    parser.add_argument(
        "--proxy-file",
        default=os.environ.get("PROXY_FILE", "proxies.txt"),
        help="Path to proxy list file (one proxy per line).",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=float(os.environ.get("REQUEST_DELAY", "1.0")),
        help="Seconds to wait between HTTP requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("MAX_RETRIES", "5")),
        help="Max retries per HTTP request.",
    )
    parser.add_argument(
        "--min-page-chars",
        type=int,
        default=200,
        help="Skip pages with fewer cleaned characters than this.",
    )
    parser.add_argument(
        "--limit-pages-per-book",
        type=int,
        default=None,
        help="Optional cap on pages processed per book (for testing).",
    )
    parser.add_argument(
        "--generate-qa",
        action="store_true",
        help="Run local Qwen model to generate Q&A (requires GPU + torch).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from pipeline_progress.json.",
    )
    parser.add_argument(
        "--test-fetch",
        action="store_true",
        help="Fetch and parse only the first book with a 2-page limit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    proxies = load_proxy_list(args.proxy_file)
    if proxies:
        print(f"Loaded {len(proxies)} proxy/proxies.")
    else:
        print("No proxies configured — using direct connection (set PROXY_LIST or proxies.txt if blocked).")

    session = ProxySession(
        proxies=proxies,
        max_retries=args.max_retries,
        request_delay=args.request_delay,
    )

    book_ids = args.book_ids
    limit_pages = args.limit_pages_per_book
    if args.test_fetch:
        book_ids = [MY_50_BOOK_IDS[0]]
        limit_pages = 2
        print(f"Test fetch mode: book {book_ids[0]}, max 2 pages.")

    generate_qa_offline = None
    if args.generate_qa:
        generate_qa_offline = load_qa_generator()

    run_pipeline(
        book_ids=book_ids,
        session=session,
        generate_qa_offline=generate_qa_offline,
        min_page_chars=args.min_page_chars,
        limit_pages_per_book=limit_pages,
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
