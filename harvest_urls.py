"""
Harvest professor profile URLs from U15 university directories.

Usage:
    uv run python harvest_urls.py                  # All schools
    uv run python harvest_urls.py --school waterloo # Single school
    uv run python harvest_urls.py --list            # List available schools
"""

import argparse
import importlib
from pathlib import Path

from scrapers.utils import save_urls_to_csv

# Registry of school scrapers: name -> module path
SCHOOLS = {
    "waterloo": "scrapers.waterloo",
    "queens": "scrapers.queens",
}


def get_scraper(school_name: str):
    return importlib.import_module(SCHOOLS[school_name])


def main():
    parser = argparse.ArgumentParser(description="Harvest professor profile URLs")
    parser.add_argument("--school", type=str, help="Scrape a single school")
    parser.add_argument("--list", action="store_true", help="List available schools")
    parser.add_argument(
        "--output", type=str, default="professor_urls.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    if args.list:
        print("Available schools:")
        for name in sorted(SCHOOLS):
            print(f"  {name}")
        return

    schools_to_scrape = [args.school] if args.school else list(SCHOOLS.keys())

    for school in schools_to_scrape:
        if school not in SCHOOLS:
            print(f"Unknown school: {school}. Use --list to see available schools.")
            return
        print(f"Harvesting {school}...")
        scraper = get_scraper(school)
        urls = scraper.harvest_all()

        # Save per-school CSV (e.g. professor_urls_waterloo.csv)
        # Use explicit --output if provided, otherwise auto-name per school
        if args.output != "professor_urls.csv" or len(schools_to_scrape) == 1:
            output_path = Path(args.output) if args.output != "professor_urls.csv" else Path(f"professor_urls_{school}.csv")
        else:
            output_path = Path(f"professor_urls_{school}.csv")
        save_urls_to_csv(urls, output_path)
        print(f"  Saved {len(urls)} URLs to {output_path}")


if __name__ == "__main__":
    main()
