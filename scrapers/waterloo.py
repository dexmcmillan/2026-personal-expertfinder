"""
University of Waterloo faculty scraper.

Directory structure:
- No central directory. Faculty organized by department.
- Listing pages: /about/people, /contacts
- Rich profile pages: /{dept-slug}/profiles/{name} or /{dept-slug}/people-profiles/{name}
- Contacts pages are thin (just name/email). Profile pages have full bios + research interests.
- Strategy: scrape listing pages for links, prefer /profiles/ URLs for extraction.
"""

import re

import requests
from bs4 import BeautifulSoup

from scrapers.utils import polite_delay

SCHOOL_NAME = "University of Waterloo"
BASE_URL = "https://uwaterloo.ca"

# Department slugs organized by faculty.
# Each entry is (slug, people_path) where people_path is the contacts/people page suffix.
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
    ],
    "Faculty of Science": [
        "biology",
        "chemistry",
        "earth-environmental-sciences",
        "physics-astronomy",
    ],
}


def _is_profile_link(href: str, dept_slug: str) -> bool:
    """Check if a link looks like a professor profile URL for this department."""
    if not href:
        return False
    patterns = [
        f"/{dept_slug}/profiles/",
        f"/{dept_slug}/people-profiles/",
        f"/{dept_slug}/about/people/",
        f"/{dept_slug}/contacts/",
        f"/{dept_slug}/profile/",
    ]
    return any(p in href for p in patterns)


def _normalize_url(href: str) -> str:
    """Ensure URL is absolute."""
    if href.startswith("http"):
        return href
    return f"{BASE_URL}{href}"


def _is_rich_profile(url: str) -> bool:
    """Check if URL points to a rich profile page (vs thin contacts page)."""
    return "/profiles/" in url or "/people-profiles/" in url


def harvest_department(dept_slug: str, faculty: str) -> list[tuple[str, str, str]]:
    """Scrape a single department's people/contacts pages for professor profile URLs.

    Prefers /profiles/ URLs (rich bios) over /contacts/ URLs (thin).
    """
    seen_urls = set()
    rich_urls = []  # /profiles/ and /people-profiles/ URLs
    thin_urls = []  # /contacts/ and /about/people/ URLs

    # Try multiple known page patterns for faculty listings
    page_paths = [
        f"/{dept_slug}/about/people",
        f"/{dept_slug}/contacts",
    ]

    for path in page_paths:
        url = f"{BASE_URL}{path}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
        except Exception:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if _is_profile_link(href, dept_slug):
                profile_url = _normalize_url(href)
                # Skip listing pages themselves
                if profile_url.rstrip("/") in (f"{BASE_URL}{p}" for p in page_paths):
                    continue
                if profile_url not in seen_urls:
                    seen_urls.add(profile_url)
                    entry = (SCHOOL_NAME, faculty, profile_url)
                    if _is_rich_profile(profile_url):
                        rich_urls.append(entry)
                    else:
                        thin_urls.append(entry)

        polite_delay(1.0)

    # Prefer rich profile URLs; fall back to thin if no rich ones found
    return rich_urls if rich_urls else thin_urls


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
