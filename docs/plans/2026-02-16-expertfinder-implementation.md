# Expert Finder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pipeline that scrapes Canadian U15 professor profiles, extracts expertise via Gemini, and provides semantic search + visual cluster exploration.

**Architecture:** Three-stage pipeline (harvest URLs → extract profiles via Gemini → embed). SQLite for structured data, numpy for embeddings, Jupyter for search/visualization. Per-school scraper modules with shared utilities.

**Tech Stack:** Python 3.12, uv, Playwright, BeautifulSoup4, google-generativeai, sentence-transformers, UMAP, Plotly, pandas, scikit-learn.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `scrapers/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Initialize uv project**

```bash
cd /Users/DMcMillan@globeandmail.com/Documents/Code/2026-personal-expertfinder
uv init --no-readme
echo "3.12" > .python-version
```

**Step 2: Add dependencies**

```bash
uv add requests beautifulsoup4 playwright pandas google-generativeai sentence-transformers umap-learn plotly scikit-learn numpy
uv add --dev pytest
```

**Step 3: Install Playwright browsers**

```bash
uv run playwright install chromium
```

**Step 4: Create .gitignore**

```
# Data files
data/
*.db
*.csv
*.npy
professor_urls.csv
professors.db
embeddings.npy

# API keys
*api_key*.txt
gemini_api_key.txt

# Python
__pycache__/
*.pyc
.venv/

# Jupyter
.ipynb_checkpoints/

# uv
uv.lock

# OS
.DS_Store
```

**Step 5: Create package directories**

```bash
mkdir -p scrapers tests
touch scrapers/__init__.py tests/__init__.py
```

**Step 6: Commit**

```bash
git add pyproject.toml .python-version .gitignore scrapers/__init__.py tests/__init__.py
git commit -m "feat: initialize project with dependencies"
```

---

### Task 2: Database Module

**Files:**
- Create: `db.py`
- Create: `tests/test_db.py`

**Step 1: Write failing tests for database operations**

Create `tests/test_db.py`:

```python
import sqlite3
import json
import tempfile
from pathlib import Path

import pytest

from db import init_db, insert_professor, get_professor_by_url, get_all_professors, get_professors_without_embeddings, update_embedding


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


def test_init_db_creates_table(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='professors'")
    assert cursor.fetchone() is not None
    conn.close()


def test_insert_and_retrieve_professor(db_path):
    prof = {
        "name": "Jane Smith",
        "title": "Associate Professor",
        "school": "University of Waterloo",
        "faculty": "Faculty of Engineering",
        "department": "Electrical and Computer Engineering",
        "email": "jsmith@uwaterloo.ca",
        "phone": "519-555-0100",
        "profile_url": "https://uwaterloo.ca/ece/profile/jsmith",
        "expertise_raw": "Machine learning, computer vision, and neural networks.",
        "expertise_keywords": ["machine learning", "computer vision", "neural networks"],
    }
    insert_professor(db_path, prof)
    result = get_professor_by_url(db_path, prof["profile_url"])
    assert result is not None
    assert result["name"] == "Jane Smith"
    assert result["school"] == "University of Waterloo"
    assert json.loads(result["expertise_keywords"]) == ["machine learning", "computer vision", "neural networks"]


def test_insert_duplicate_url_skips(db_path):
    prof = {
        "name": "Jane Smith",
        "title": "Associate Professor",
        "school": "University of Waterloo",
        "faculty": "Faculty of Engineering",
        "department": "ECE",
        "email": None,
        "phone": None,
        "profile_url": "https://uwaterloo.ca/ece/profile/jsmith",
        "expertise_raw": "ML research.",
        "expertise_keywords": ["ML"],
    }
    insert_professor(db_path, prof)
    insert_professor(db_path, prof)  # should not raise
    all_profs = get_all_professors(db_path)
    assert len(all_profs) == 1


def test_get_professors_without_embeddings(db_path):
    prof = {
        "name": "Jane Smith",
        "title": "Professor",
        "school": "UW",
        "faculty": "Eng",
        "department": "ECE",
        "email": None,
        "phone": None,
        "profile_url": "https://example.com/jsmith",
        "expertise_raw": "AI research.",
        "expertise_keywords": ["AI"],
    }
    insert_professor(db_path, prof)
    without = get_professors_without_embeddings(db_path)
    assert len(without) == 1
    assert without[0]["name"] == "Jane Smith"


def test_update_embedding(db_path):
    import numpy as np

    prof = {
        "name": "Jane Smith",
        "title": "Professor",
        "school": "UW",
        "faculty": "Eng",
        "department": "ECE",
        "email": None,
        "phone": None,
        "profile_url": "https://example.com/jsmith",
        "expertise_raw": "AI research.",
        "expertise_keywords": ["AI"],
    }
    insert_professor(db_path, prof)
    embedding = np.random.rand(384).astype(np.float32)
    update_embedding(db_path, prof["profile_url"], embedding)
    result = get_professor_by_url(db_path, prof["profile_url"])
    assert result["embedding"] is not None
    restored = np.frombuffer(result["embedding"], dtype=np.float32)
    assert restored.shape == (384,)
    assert np.allclose(restored, embedding)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_db.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'db'`

**Step 3: Implement db.py**

Create `db.py`:

```python
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path("professors.db")


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS professors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            title TEXT,
            school TEXT NOT NULL,
            faculty TEXT,
            department TEXT,
            email TEXT,
            phone TEXT,
            profile_url TEXT UNIQUE NOT NULL,
            expertise_raw TEXT,
            expertise_keywords TEXT,
            embedding BLOB,
            scraped_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_professor(db_path: Path, prof: dict) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """INSERT OR IGNORE INTO professors
            (name, title, school, faculty, department, email, phone,
             profile_url, expertise_raw, expertise_keywords, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prof["name"],
                prof.get("title"),
                prof["school"],
                prof.get("faculty"),
                prof.get("department"),
                prof.get("email"),
                prof.get("phone"),
                prof["profile_url"],
                prof.get("expertise_raw"),
                json.dumps(prof.get("expertise_keywords", [])),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_professor_by_url(db_path: Path, url: str) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT * FROM professors WHERE profile_url = ?", (url,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_professors(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM professors").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_professors_without_embeddings(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM professors WHERE embedding IS NULL").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_embedding(db_path: Path, profile_url: str, embedding) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE professors SET embedding = ? WHERE profile_url = ?",
        (embedding.tobytes(), profile_url),
    )
    conn.commit()
    conn.close()
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_db.py -v
```

Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add db.py tests/test_db.py
git commit -m "feat: add database module with CRUD operations"
```

---

### Task 3: Scraper Utilities

**Files:**
- Create: `scrapers/utils.py`
- Create: `tests/test_scraper_utils.py`

**Step 1: Write failing tests**

Create `tests/test_scraper_utils.py`:

```python
import csv
import tempfile
from pathlib import Path

from scrapers.utils import clean_html, save_urls_to_csv, load_urls_from_csv


def test_clean_html_strips_nav_and_scripts():
    html = """
    <html>
    <head><script>var x = 1;</script></head>
    <body>
    <nav><a href="/">Home</a></nav>
    <header>Site Header</header>
    <main><p>Professor Jane Smith studies AI.</p></main>
    <footer>Copyright 2024</footer>
    </body>
    </html>
    """
    cleaned = clean_html(html)
    assert "Professor Jane Smith studies AI." in cleaned
    assert "var x = 1" not in cleaned
    assert "Site Header" not in cleaned
    assert "Copyright" not in cleaned


def test_save_and_load_urls(tmp_path):
    urls = [
        ("University of Waterloo", "Faculty of Engineering", "https://example.com/prof1"),
        ("University of Waterloo", "Faculty of Science", "https://example.com/prof2"),
    ]
    csv_path = tmp_path / "urls.csv"
    save_urls_to_csv(urls, csv_path)
    loaded = load_urls_from_csv(csv_path)
    assert len(loaded) == 2
    assert loaded[0] == {"school": "University of Waterloo", "faculty": "Faculty of Engineering", "url": "https://example.com/prof1"}
    assert loaded[1] == {"school": "University of Waterloo", "faculty": "Faculty of Science", "url": "https://example.com/prof2"}
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_scraper_utils.py -v
```

Expected: FAIL

**Step 3: Implement scrapers/utils.py**

Create `scrapers/utils.py`:

```python
import csv
import time
from pathlib import Path

from bs4 import BeautifulSoup


def clean_html(html: str) -> str:
    """Strip navigation, scripts, headers, footers — return just the main content text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def polite_delay(seconds: float = 2.0) -> None:
    """Sleep between requests to be respectful to servers."""
    time.sleep(seconds)


def save_urls_to_csv(urls: list[tuple[str, str, str]], path: Path) -> None:
    """Save (school, faculty, url) tuples to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["school", "faculty", "url"])
        writer.writerows(urls)


def load_urls_from_csv(path: Path) -> list[dict]:
    """Load professor URLs from CSV."""
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_scraper_utils.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add scrapers/utils.py tests/test_scraper_utils.py
git commit -m "feat: add scraper utilities (HTML cleaning, CSV I/O)"
```

---

### Task 4: Gemini Extraction Module

**Files:**
- Create: `gemini_extract.py`
- Create: `tests/test_gemini_extract.py`

**Step 1: Write failing tests (mocking Gemini)**

Create `tests/test_gemini_extract.py`:

```python
import json
from unittest.mock import patch, MagicMock

from gemini_extract import build_extraction_prompt, parse_gemini_response, extract_profile


def test_build_extraction_prompt_includes_html():
    prompt = build_extraction_prompt("<p>Dr. Smith studies AI ethics.</p>")
    assert "Dr. Smith studies AI ethics." in prompt
    assert "name" in prompt.lower()
    assert "expertise" in prompt.lower()


def test_parse_gemini_response_valid_json():
    response_text = json.dumps({
        "name": "Jane Smith",
        "title": "Associate Professor",
        "department": "Computer Science",
        "email": "jsmith@uwaterloo.ca",
        "phone": None,
        "expertise_raw": "Machine learning and computer vision research.",
        "expertise_keywords": ["machine learning", "computer vision"],
    })
    result = parse_gemini_response(response_text)
    assert result["name"] == "Jane Smith"
    assert result["expertise_keywords"] == ["machine learning", "computer vision"]


def test_parse_gemini_response_json_in_markdown():
    response_text = '```json\n{"name": "Jane Smith", "title": "Prof", "department": "CS", "email": null, "phone": null, "expertise_raw": "AI.", "expertise_keywords": ["AI"]}\n```'
    result = parse_gemini_response(response_text)
    assert result["name"] == "Jane Smith"


def test_parse_gemini_response_invalid_returns_none():
    result = parse_gemini_response("This is not JSON at all.")
    assert result is None
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_gemini_extract.py -v
```

Expected: FAIL

**Step 3: Implement gemini_extract.py**

Create `gemini_extract.py`:

```python
import json
import re
from pathlib import Path

import google.generativeai as genai

from scrapers.utils import clean_html


def load_gemini_key(path: Path = Path("gemini_api_key.txt")) -> str:
    return path.read_text().strip()


def configure_gemini(api_key: str | None = None) -> None:
    key = api_key or load_gemini_key()
    genai.configure(api_key=key)


def build_extraction_prompt(html_text: str) -> str:
    return f"""Extract the following fields from this professor's bio page. Return ONLY valid JSON with these fields:

- "name": full name (string)
- "title": academic title like "Associate Professor", "Professor", "Assistant Professor" (string or null)
- "department": department name (string or null)
- "email": email address (string or null)
- "phone": phone number (string or null)
- "expertise_raw": a 1-3 sentence summary of their research interests and expertise (string)
- "expertise_keywords": a list of 3-10 specific expertise keywords/phrases (list of strings)

If a field is not found, use null. For expertise_keywords, extract specific research topics, not generic terms.

Bio page content:
{html_text}"""


def parse_gemini_response(response_text: str) -> dict | None:
    text = response_text.strip()
    # Try to extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_profile(html: str, model_name: str = "gemini-2.0-flash") -> dict | None:
    cleaned = clean_html(html)
    prompt = build_extraction_prompt(cleaned)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return parse_gemini_response(response.text)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_gemini_extract.py -v
```

Expected: All PASS (tests only exercise prompt building and response parsing, no API calls).

**Step 5: Commit**

```bash
git add gemini_extract.py tests/test_gemini_extract.py
git commit -m "feat: add Gemini extraction module with prompt and parser"
```

---

### Task 5: First School Scraper — University of Waterloo

**Files:**
- Create: `scrapers/waterloo.py`

This is the pilot scraper to prove the pattern. Waterloo is a good starting point because:
- No pagination (all faculty listed on one page per department)
- Minimal JavaScript needed for contact listings
- Clean URL pattern: `/[dept-slug]/profile/[username]`

**Step 1: Investigate Waterloo's directory structure**

Manually browse https://uwaterloo.ca/faculties-academics to identify all department slugs. The scraper needs a list of department URL slugs.

**Step 2: Implement scrapers/waterloo.py**

Create `scrapers/waterloo.py`:

```python
"""
University of Waterloo faculty scraper.

Directory structure:
- No central directory. Faculty organized by department.
- Contact pages: https://uwaterloo.ca/{dept-slug}/contacts?title=&group[61]=61
- Profile pages: https://uwaterloo.ca/{dept-slug}/profile/{username}
- No pagination — all faculty on one page per department.
"""

import re

import requests
from bs4 import BeautifulSoup

from scrapers.utils import polite_delay

SCHOOL_NAME = "University of Waterloo"

# Department slugs organized by faculty.
# This list needs to be built by browsing https://uwaterloo.ca/faculties-academics
# and collecting each department's URL slug.
DEPARTMENTS = {
    "Faculty of Arts": [
        "anthropology",
        "classical-studies",
        "communication-arts",
        "economics",
        "english",
        "fine-arts",
        "french-studies",
        "germanic-slavic-studies",
        "history",
        "philosophy",
        "political-science",
        "psychology",
        "religious-studies",
        "sociology-and-legal-studies",
        "spanish-and-latin-american-studies",
    ],
    "Faculty of Engineering": [
        "architecture",
        "chemical-engineering",
        "civil-environmental-engineering",
        "electrical-computer-engineering",
        "management-science-engineering",
        "mechanical-mechatronics-engineering",
        "systems-design-engineering",
    ],
    "Faculty of Environment": [
        "environment-enterprise-development",
        "geography-environmental-management",
        "knowledge-integration",
        "planning",
        "school-of-environment-resources-and-sustainability",
    ],
    "Faculty of Health": [
        "kinesiology-health-sciences",
        "pharmacy",
        "public-health-sciences",
        "recreation-leisure-studies",
        "school-of-optometry-vision-science",
    ],
    "Faculty of Mathematics": [
        "applied-mathematics",
        "combinatorics-and-optimization",
        "computer-science",
        "pure-mathematics",
        "statistics-and-actuarial-science",
        "david-r-cheriton-school-of-computer-science",
    ],
    "Faculty of Science": [
        "biology",
        "chemistry",
        "earth-environmental-sciences",
        "physics-astronomy",
    ],
}


def harvest_department(dept_slug: str, faculty: str) -> list[tuple[str, str, str]]:
    """Scrape a single department's contacts page for professor profile URLs."""
    url = f"https://uwaterloo.ca/{dept_slug}/contacts?title=&group[61]=61"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    # Look for links that match the profile URL pattern
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if f"/{dept_slug}/profile/" in href or re.match(r"/[^/]+/profile/", href):
            profile_url = href if href.startswith("http") else f"https://uwaterloo.ca{href}"
            if profile_url not in [r[2] for r in results]:
                results.append((SCHOOL_NAME, faculty, profile_url))

    return results


def harvest_all() -> list[tuple[str, str, str]]:
    """Scrape all Waterloo departments for professor profile URLs."""
    all_urls = []
    for faculty, depts in DEPARTMENTS.items():
        for dept_slug in depts:
            print(f"  Scraping {dept_slug}...")
            try:
                urls = harvest_department(dept_slug, faculty)
                all_urls.extend(urls)
                print(f"    Found {len(urls)} profiles")
            except Exception as e:
                print(f"    ERROR: {e}")
            polite_delay(2.0)
    return all_urls
```

**Step 3: Test manually**

```bash
uv run python -c "from scrapers.waterloo import harvest_department; print(harvest_department('computer-science', 'Faculty of Mathematics')[:3])"
```

Verify it returns a list of (school, faculty, url) tuples. The department slugs may need adjustment based on actual site structure — this is expected iteration.

**Step 4: Commit**

```bash
git add scrapers/waterloo.py
git commit -m "feat: add University of Waterloo scraper"
```

---

### Task 6: harvest_urls.py CLI Script

**Files:**
- Create: `harvest_urls.py`

**Step 1: Implement harvest_urls.py**

Create `harvest_urls.py`:

```python
"""
Harvest professor profile URLs from U15 university directories.

Usage:
    uv run python harvest_urls.py                  # All schools
    uv run python harvest_urls.py --school waterloo # Single school
    uv run python harvest_urls.py --list            # List available schools
"""

import argparse
from pathlib import Path

from scrapers.utils import save_urls_to_csv

# Registry of school scrapers
SCHOOLS = {}


def register_school(name, module_path):
    SCHOOLS[name] = module_path


# Register available scrapers
register_school("waterloo", "scrapers.waterloo")


def get_scraper(school_name: str):
    import importlib
    module = importlib.import_module(SCHOOLS[school_name])
    return module


def main():
    parser = argparse.ArgumentParser(description="Harvest professor profile URLs")
    parser.add_argument("--school", type=str, help="Scrape a single school")
    parser.add_argument("--list", action="store_true", help="List available schools")
    parser.add_argument("--output", type=str, default="professor_urls.csv", help="Output CSV path")
    args = parser.parse_args()

    if args.list:
        print("Available schools:")
        for name in sorted(SCHOOLS):
            print(f"  {name}")
        return

    schools_to_scrape = [args.school] if args.school else list(SCHOOLS.keys())
    all_urls = []

    for school in schools_to_scrape:
        if school not in SCHOOLS:
            print(f"Unknown school: {school}. Use --list to see available schools.")
            return
        print(f"Harvesting {school}...")
        scraper = get_scraper(school)
        urls = scraper.harvest_all()
        all_urls.extend(urls)
        print(f"  Total: {len(urls)} URLs")

    output_path = Path(args.output)
    save_urls_to_csv(all_urls, output_path)
    print(f"\nSaved {len(all_urls)} URLs to {output_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Test the CLI**

```bash
uv run python harvest_urls.py --list
```

Expected: Shows "waterloo" as available school.

**Step 3: Commit**

```bash
git add harvest_urls.py
git commit -m "feat: add harvest_urls.py CLI script"
```

---

### Task 7: extract_profiles.py Pipeline Script

**Files:**
- Create: `extract_profiles.py`

**Step 1: Implement extract_profiles.py**

Create `extract_profiles.py`:

```python
"""
Extract professor profiles from harvested URLs using Gemini.

Usage:
    uv run python extract_profiles.py                          # Process all
    uv run python extract_profiles.py --input urls.csv         # Custom input
    uv run python extract_profiles.py --limit 10               # Process first 10 only
    uv run python extract_profiles.py --school "University of Waterloo"  # Filter by school
"""

import argparse
from pathlib import Path

import requests

from db import init_db, insert_professor, get_professor_by_url, DEFAULT_DB_PATH
from gemini_extract import configure_gemini, extract_profile
from scrapers.utils import load_urls_from_csv, clean_html, polite_delay


def fetch_page(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  Failed to fetch {url}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract professor profiles via Gemini")
    parser.add_argument("--input", type=str, default="professor_urls.csv", help="Input CSV of URLs")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of profiles to process (0=all)")
    parser.add_argument("--school", type=str, help="Only process URLs from this school")
    args = parser.parse_args()

    db_path = Path(args.db)
    init_db(db_path)
    configure_gemini()

    urls = load_urls_from_csv(Path(args.input))
    if args.school:
        urls = [u for u in urls if u["school"] == args.school]
    if args.limit:
        urls = urls[: args.limit]

    print(f"Processing {len(urls)} URLs...")
    success, skipped, failed = 0, 0, 0

    for i, entry in enumerate(urls):
        url = entry["url"]
        school = entry["school"]
        faculty = entry["faculty"]

        # Checkpoint: skip if already in DB
        if get_professor_by_url(db_path, url):
            skipped += 1
            continue

        print(f"[{i+1}/{len(urls)}] {url}")
        html = fetch_page(url)
        if not html:
            failed += 1
            polite_delay(2.0)
            continue

        profile = extract_profile(html)
        if not profile:
            print(f"  Gemini extraction failed")
            failed += 1
            polite_delay(2.0)
            continue

        # Merge scraped metadata with extracted data
        profile["school"] = school
        profile["faculty"] = faculty
        profile["profile_url"] = url
        insert_professor(db_path, profile)
        success += 1
        polite_delay(2.0)

    print(f"\nDone. Success: {success}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add extract_profiles.py
git commit -m "feat: add extract_profiles.py Gemini extraction pipeline"
```

---

### Task 8: embed_profiles.py

**Files:**
- Create: `embed_profiles.py`
- Create: `tests/test_embed.py`

**Step 1: Write failing test**

Create `tests/test_embed.py`:

```python
import numpy as np

from embed_profiles import embed_texts


def test_embed_texts_returns_correct_shape():
    texts = ["Machine learning and AI research", "Climate change policy analysis"]
    embeddings = embed_texts(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 output dim


def test_embed_texts_similar_texts_closer():
    texts = [
        "Machine learning and deep neural networks",
        "Artificial intelligence and deep learning",
        "Medieval French poetry and literature",
    ]
    embeddings = embed_texts(texts)
    # ML topics should be closer to each other than to poetry
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(embeddings)
    assert sims[0, 1] > sims[0, 2]
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_embed.py -v
```

Expected: FAIL

**Step 3: Implement embed_profiles.py**

Create `embed_profiles.py`:

```python
"""
Generate embeddings for professor expertise text.

Usage:
    uv run python embed_profiles.py                    # Embed all without embeddings
    uv run python embed_profiles.py --db professors.db # Custom DB path
"""

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from db import (
    get_professors_without_embeddings,
    get_all_professors,
    update_embedding,
    DEFAULT_DB_PATH,
)

MODEL_NAME = "all-MiniLM-L6-v2"


def embed_texts(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True)


def main():
    parser = argparse.ArgumentParser(description="Generate expertise embeddings")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--output", type=str, default="embeddings.npy", help="Output numpy file")
    args = parser.parse_args()

    db_path = Path(args.db)

    # Embed professors that don't have embeddings yet
    profs = get_professors_without_embeddings(db_path)
    if profs:
        print(f"Embedding {len(profs)} professors...")
        texts = [p["expertise_raw"] or "" for p in profs]
        embeddings = embed_texts(texts)
        for prof, emb in zip(profs, embeddings):
            update_embedding(db_path, prof["profile_url"], emb.astype(np.float32))
        print(f"Updated {len(profs)} embeddings in DB.")

    # Export all embeddings as numpy array
    all_profs = get_all_professors(db_path)
    all_with_emb = [p for p in all_profs if p["embedding"] is not None]
    if all_with_emb:
        matrix = np.array(
            [np.frombuffer(p["embedding"], dtype=np.float32) for p in all_with_emb]
        )
        np.save(args.output, matrix)
        print(f"Saved {matrix.shape[0]} embeddings ({matrix.shape[1]}d) to {args.output}")
    else:
        print("No embeddings to export.")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_embed.py -v
```

Expected: All PASS (note: first run will download the model, ~80MB).

**Step 5: Commit**

```bash
git add embed_profiles.py tests/test_embed.py
git commit -m "feat: add embedding generation with sentence-transformers"
```

---

### Task 9: Search & Visualization Notebook

**Files:**
- Create: `search.ipynb`

**Step 1: Create the Jupyter notebook with the following cells:**

**Cell 1 — Imports and setup:**

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import umap

from db import get_all_professors

DB_PATH = Path("professors.db")
MODEL_NAME = "all-MiniLM-L6-v2"

# Load data
profs = get_all_professors(DB_PATH)
df = pd.DataFrame(profs)
df["expertise_keywords"] = df["expertise_keywords"].apply(lambda x: json.loads(x) if x else [])
df["keywords_str"] = df["expertise_keywords"].apply(lambda x: ", ".join(x))

# Load embeddings from DB
embeddings = np.array([
    np.frombuffer(row["embedding"], dtype=np.float32)
    for _, row in df.iterrows()
    if row["embedding"] is not None
])

# Load model for query embedding
model = SentenceTransformer(MODEL_NAME)

print(f"Loaded {len(df)} professors with {embeddings.shape[0]} embeddings")
```

**Cell 2 — Semantic search function:**

```python
def search_experts(query: str, top_n: int = 20, school: str = None, faculty: str = None):
    """Search for professors by natural language query."""
    mask = pd.Series([True] * len(df))
    if school:
        mask &= df["school"] == school
    if faculty:
        mask &= df["faculty"] == faculty

    filtered_df = df[mask].reset_index(drop=True)
    filtered_emb = embeddings[mask.values]

    query_emb = model.encode([query])
    sims = cosine_similarity(query_emb, filtered_emb)[0]

    filtered_df["similarity"] = sims
    results = filtered_df.nlargest(top_n, "similarity")
    return results[["name", "school", "faculty", "department", "keywords_str", "email", "similarity"]]
```

**Cell 3 — Example search:**

```python
search_experts("climate change policy and economics")
```

**Cell 4 — UMAP visualization:**

```python
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
coords = reducer.fit_transform(embeddings)
df["umap_x"] = coords[:, 0]
df["umap_y"] = coords[:, 1]

fig = px.scatter(
    df, x="umap_x", y="umap_y",
    color="school",
    hover_data=["name", "faculty", "department", "keywords_str"],
    title="Canadian University Professors by Expertise",
    width=1200, height=800,
)
fig.update_traces(marker=dict(size=4, opacity=0.7))
fig.show()
```

**Cell 5 — Color by faculty instead:**

```python
fig2 = px.scatter(
    df, x="umap_x", y="umap_y",
    color="faculty",
    hover_data=["name", "school", "department", "keywords_str"],
    title="Canadian University Professors by Faculty",
    width=1200, height=800,
)
fig2.update_traces(marker=dict(size=4, opacity=0.7))
fig2.show()
```

**Step 2: Commit**

```bash
git add search.ipynb
git commit -m "feat: add search and visualization notebook"
```

---

### Task 10: End-to-End Test with Waterloo (Pilot Run)

This is a manual integration test — run the full pipeline for a single department to verify everything works end to end.

**Step 1: Run harvest for one department**

```bash
uv run python -c "
from scrapers.waterloo import harvest_department
from scrapers.utils import save_urls_to_csv
urls = harvest_department('computer-science', 'Faculty of Mathematics')
save_urls_to_csv(urls, 'professor_urls.csv')
print(f'Harvested {len(urls)} URLs')
"
```

**Step 2: Run extraction on a small sample**

```bash
uv run python extract_profiles.py --limit 5
```

Verify output: should see 5 professors extracted and inserted into `professors.db`.

**Step 3: Run embedding**

```bash
uv run python embed_profiles.py
```

**Step 4: Open notebook and test search**

```bash
uv run jupyter notebook search.ipynb
```

Run the first 3 cells. Verify search returns results.

**Step 5: Iterate and fix**

Fix any issues with the Waterloo scraper (department slugs, URL patterns, etc.). Once one department works, run the full Waterloo harvest:

```bash
uv run python harvest_urls.py --school waterloo
uv run python extract_profiles.py --limit 50
uv run python embed_profiles.py
```

**Step 6: Commit any fixes**

```bash
git add -A
git commit -m "fix: adjust waterloo scraper after pilot run"
```

---

### Task 11+: Additional School Scrapers (Future)

After the Waterloo pilot proves the pattern, add scrapers for the remaining 14 U15 schools. Each follows the same pattern:

1. Create `scrapers/{school}.py` with `DEPARTMENTS` dict and `harvest_all()` function
2. Register in `harvest_urls.py`
3. Test with a single department first
4. Run full harvest

Priority order (easiest to hardest based on directory research):
1. Queen's — no pagination, server-rendered, clean HTML
2. McMaster — engineering faculty is clean, others vary
3. UBC — department-organized, relatively consistent
4. Alberta — browsable directory
5. Saskatchewan, Manitoba, Dalhousie, Calgary, Western — department-based, similar patterns
6. McGill — central directory with pagination
7. Ottawa — bilingual
8. UofT — complex, multiple sub-directories per department
9. Laval, Université de Montréal — French-language

Each scraper is an independent task that follows the same module pattern.
