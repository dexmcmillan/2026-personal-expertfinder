import csv
import time
from pathlib import Path

from bs4 import BeautifulSoup


def clean_html(html: str) -> str:
    """Strip navigation, scripts, headers, footers — return just the main content text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(
        ["script", "style", "nav", "header", "footer", "noscript"]
    ):
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
