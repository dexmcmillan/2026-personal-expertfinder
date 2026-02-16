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
from scrapers.utils import load_urls_from_csv, polite_delay


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
    parser.add_argument(
        "--input", type=str, default="professor_urls.csv", help="Input CSV of URLs"
    )
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB_PATH), help="SQLite database path"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of profiles to process (0=all)"
    )
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

        print(f"[{i + 1}/{len(urls)}] {url}")
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
        print(f"  -> {profile.get('name', 'Unknown')} ({profile.get('title', '')})")
        polite_delay(2.0)

    print(f"\nDone. Success: {success}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
