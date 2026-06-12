# QuestionAi — Ketabonline Q&A Dataset Pipeline

Fetches 50 Islamic jurisprudence books from [ketabonline.com](https://ketabonline.com), parses each page, and generates Arabic Q&A pairs for AI model training.

## Problem: 403 blocks

Direct API requests to `backend.ketabonline.com` may return **403 Forbidden**. This pipeline uses **proxy rotation** with retries, browser-like headers, and request throttling to work around blocks.

## Setup

```bash
pip install -r requirements-fetch.txt   # fetch/parse only
# OR for full Q&A generation on GPU (Colab):
pip install -r requirements.txt
```

### Configure proxies

Copy the example and add your working proxies:

```bash
cp proxies.example.txt proxies.txt
# Edit proxies.txt — one proxy per line
```

Or use environment variables:

```bash
export PROXY_LIST="http://proxy1:8080,http://proxy2:8080"
export REQUEST_DELAY=1.5
export MAX_RETRIES=8
```

## Usage

### Test fetch (1 book, 2 pages)

```bash
python generate_qa_dataset.py --test-fetch
```

### Fetch & parse all 50 books (no GPU needed)

```bash
python generate_qa_dataset.py
```

Parsed books are saved to `data/book_<id>.json`. Progress resumes automatically via `pipeline_progress.json`.

### Full Q&A generation

**From cached data** (after fetch is done):

```bash
python3 generate_qa_from_cache.py
```

- **No GPU:** auto-uses `Qwen2.5-0.5B-Instruct` on CPU (~13s/page, ~7–9 days for 51k pages)
- **With GPU:** auto-uses `Qwen2.5-7B-Instruct` (4-bit)
- Resumes via `qa_progress.json` — safe to stop/restart
- Output: `generated_questions_dataset.json`

**Fetch + Q&A in one step** (GPU / Colab):

```bash
python3 generate_qa_dataset.py --generate-qa
```

### Options

| Flag | Description |
|------|-------------|
| `--proxy-file proxies.txt` | Proxy list file |
| `--request-delay 1.0` | Seconds between requests |
| `--max-retries 5` | Retries per request (rotates proxy each time) |
| `--limit-pages-per-book N` | Cap pages per book (testing) |
| `--book-ids 5171 10653` | Process specific books only |
| `--no-resume` | Start from scratch |
| `--generate-qa` | Enable local Qwen Q&A generation |

## Output format

Each Q&A entry in `generated_questions_dataset.json`:

```json
{
  "question": "...",
  "answer": "...",
  "reason": "definition",
  "reference": {
    "book_id": 5171,
    "book_title": "أصول السرخسي",
    "page_number": 12,
    "part_name": "1",
    "breadcrumbs": ["المقدمة", "..."]
  }
}
```

## 50 book IDs

The default list covers 50 usul al-fiqh books (IDs hardcoded in `generate_qa_dataset.py`).
