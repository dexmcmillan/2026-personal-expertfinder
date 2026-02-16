"""
University of Waterloo faculty scraper.

Directory structure:
- No central directory. Faculty organized by department.
- Contact/people pages vary by department:
  - /about/people (common)
  - /contacts (common)
- Profile URLs: /{dept-slug}/about/people/{username} or /{dept-slug}/contacts/{username}
- No pagination — all faculty typically on one page per department.
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
    # Match patterns like /dept/about/people/username or /dept/contacts/username
    patterns = [
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


def harvest_department(dept_slug: str, faculty: str) -> list[tuple[str, str, str]]:
    """Scrape a single department's people/contacts pages for professor profile URLs."""
    results = []
    seen_urls = set()

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
                # Skip if it's just the listing page itself
                if profile_url.rstrip("/") in (f"{BASE_URL}{p}" for p in page_paths):
                    continue
                if profile_url not in seen_urls:
                    seen_urls.add(profile_url)
                    results.append((SCHOOL_NAME, faculty, profile_url))

        polite_delay(1.0)

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
