# Expert Finder: Design Document

## Overview

A tool that scrapes professor profiles from Canada's U15 research universities, uses Gemini to extract structured expertise data, embeds the expertise text with a local sentence transformer, and provides natural language search + visual cluster exploration via Jupyter notebook.

**Project:** `2026-personal-expertfinder`
**Cost estimate:** ~$15-20 USD in Gemini API calls. Embeddings are free (local model).

## Data Model

| Field | Type | Source |
|-------|------|--------|
| `id` | integer (auto) | Generated |
| `name` | text | Gemini extraction |
| `title` | text | Gemini (e.g. "Associate Professor") |
| `school` | text | Known (e.g. "University of Toronto") |
| `faculty` | text | Gemini (e.g. "Faculty of Medicine") |
| `department` | text | Gemini (e.g. "Department of Immunology") |
| `email` | text, nullable | Gemini |
| `phone` | text, nullable | Gemini |
| `profile_url` | text | Scraped |
| `expertise_raw` | text | Gemini - full bio/research interests text |
| `expertise_keywords` | text (JSON array) | Gemini - extracted keywords |
| `embedding` | blob (numpy) | Sentence transformer output |
| `scraped_at` | datetime | Timestamp |

Storage: SQLite for structured data. Embeddings stored as both `.npy` file (for fast numpy operations) and as blobs in SQLite (for self-contained querying).

## Pipeline Architecture

Three independent scripts run in sequence:

```
1. harvest_urls.py     →  professor_urls.csv
2. extract_profiles.py →  professors.db (SQLite)
3. embed_profiles.py   →  embeddings.npy + updates professors.db
```

### Step 1: harvest_urls.py

- One function per school that hits the faculty directory and collects profile URLs
- Each school function returns a list of `(school_name, faculty, profile_url)` tuples
- Outputs a CSV so you can inspect/edit before running extraction
- Can run per-school or all at once via CLI flag
- School-specific scraper modules live in `scrapers/` directory

### Step 2: extract_profiles.py

- Reads `professor_urls.csv`, fetches each profile page's HTML
- Strips navigation/headers to minimize token usage
- Sends cleaned HTML to Gemini with a structured prompt requesting JSON output
- Gemini returns structured fields which get inserted into SQLite
- ~2 second rate limiting between API calls
- Checkpointing: skips URLs already in the DB so interrupted runs can resume

### Step 3: embed_profiles.py

- Loads `expertise_raw` text from SQLite
- Runs through `sentence-transformers/all-MiniLM-L6-v2` (local, free, 384-dim output)
- Saves embeddings as `embeddings.npy` (matrix of shape `[n_professors, 384]`)
- Also stores embedding as blob in SQLite

## U15 School Scrapers

Each school gets its own module in `scrapers/`:

| School | Notes |
|--------|-------|
| University of Toronto | Multiple faculty directories per department |
| UBC | Department-organized, relatively consistent |
| McGill | Central directory with search/pagination |
| Alberta | Faculty & Staff browsable directory |
| Calgary | Department-based faculty pages |
| Dalhousie | Faculty listings per department |
| Laval | French-language directories |
| Manitoba | Department-based listings |
| McMaster | Department faculty pages |
| Universite de Montreal | French-language, department-based |
| Ottawa | Bilingual directories |
| Queen's | Department-based faculty pages |
| Saskatchewan | Faculty listings per college |
| Waterloo | Department-based, clean HTML |
| Western | Department-based faculty pages |

Common utilities in `scraper_utils.py`: `fetch_page()`, `polite_delay()`, CSV output helpers.

Most schools will need Playwright (JavaScript-rendered directories). Respect `robots.txt`, 2-3 second delays between requests.

## Search & Visualization (search.ipynb)

### Natural language search

- Embed query with same sentence-transformer model
- Cosine similarity against all professor embeddings
- Return top N ranked results as a table: Name, School, Faculty, Expertise Keywords, Similarity Score, Email

### Visual cluster exploration

- UMAP reduces 384-dim embeddings to 2D
- Plotly interactive scatter plot, each dot = professor
- Color by school or faculty (toggle)
- Hover shows name, school, top expertise keywords
- Zoom into clusters to discover groupings

### Filtering

- Filter by school before searching
- Filter by faculty
- Combine filters with semantic search

## Dependencies

- `requests` / `playwright` - web scraping
- `beautifulsoup4` - HTML parsing
- `google-generativeai` - Gemini API
- `sentence-transformers` - local embedding model
- `umap-learn` - dimensionality reduction
- `plotly` - interactive visualization
- `scikit-learn` - cosine similarity
- `pandas` - data manipulation

## Risks

- **Per-school scraper maintenance**: University sites change. Scrapers will break.
- **Coverage gaps**: Some professors may not have public profile pages.
- **Rate limiting**: Need to be polite to university servers. Full scrape of all 15 schools will take many hours.
- **French-language sites**: Laval, Montreal, Ottawa (partial) — Gemini handles this but keyword extraction quality may vary.
