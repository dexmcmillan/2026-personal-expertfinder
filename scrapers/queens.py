"""
Queen's University faculty scraper.

Directory structure:
- Highly heterogeneous. Each faculty/school has its own site and URL patterns.
- Most Arts & Science depts: queensu.ca/{dept}/people with profile links /{dept}/people/{name-slug}
- Some depts use subdomains: econ.queensu.ca, biology.queensu.ca, cs.queensu.ca, chem.queensu.ca
- Education: educ.queensu.ca/people (paginated)
- Law: law.queensu.ca/directory
- Health Sciences: meds.queensu.ca, nursing.queensu.ca, rehab.queensu.ca
- Smith Engineering: smithengineering.queensu.ca (JS-rendered, needs Playwright)
- Smith Business: smith.queensu.ca (403s on requests, needs Playwright)

Strategy:
- Scrape server-rendered departments with requests + BeautifulSoup
- Use Playwright for JS-rendered directories (Engineering, Business)
- Handle pagination where needed (Education)
"""

import re

import requests
from bs4 import BeautifulSoup

from scrapers.utils import polite_delay

SCHOOL_NAME = "Queen's University"
BASE_URL = "https://www.queensu.ca"

# User-Agent to identify ourselves politely
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
}


def _fetch(url: str) -> requests.Response | None:
    """Fetch a URL with error handling."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"    Failed to fetch {url}: {e}")
        return None


def _extract_profile_links(soup: BeautifulSoup, base_url: str, path_pattern: str) -> list[str]:
    """Extract profile URLs matching a path pattern from a page."""
    urls = []
    seen = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if path_pattern in href:
            # Normalize URL
            if href.startswith("http"):
                full_url = href
            elif href.startswith("/"):
                # Determine base from the base_url
                from urllib.parse import urlparse
                parsed = urlparse(base_url)
                full_url = f"{parsed.scheme}://{parsed.netloc}{href}"
            else:
                full_url = f"{base_url.rstrip('/')}/{href}"
            # Remove trailing slashes and query strings for dedup
            clean = full_url.split("?")[0].rstrip("/")
            if clean not in seen:
                seen.add(clean)
                urls.append(clean)
    return urls


# ──────────────────────────────────────────────────────────
# Arts & Science departments on queensu.ca/{dept}/people
# ──────────────────────────────────────────────────────────

# (dept_slug, people_path, profile_path_pattern)
# Most depts: /people page with profile links containing /{dept}/people/
ARTSCI_QUEENSU_DEPTS = [
    # Departments hosted at queensu.ca/{slug}/people that list profiles directly.
    # These use the standard pattern: /people page with profile links inside content (not nav).
    ("art", "Faculty of Arts and Science"),
    ("english", "Faculty of Arts and Science"),
    ("geographyandplanning", "Faculty of Arts and Science"),
    ("devs", "Faculty of Arts and Science"),
    ("history", "Faculty of Arts and Science"),
    ("philosophy", "Faculty of Arts and Science"),
    ("politics", "Faculty of Arts and Science"),
    ("psychology", "Faculty of Arts and Science"),
    ("ensc", "Faculty of Arts and Science"),
]

# queensu.ca departments with non-standard people page paths.
# Format: (base_url, people_page_path, link_prefix, faculty)
ARTSCI_SPECIAL_DEPTS = [
    # (base_url, people_page_path, link_prefix, faculty)
    # Departments with non-standard people page paths on queensu.ca
    (BASE_URL, "/mathstat/people/faculty", "/mathstat/people/faculty/profiles/", "Faculty of Arts and Science"),
    (BASE_URL, "/physics/people", "/physics/people-search/", "Faculty of Arts and Science"),
    # BFA handled by _harvest_bfa (profile links are in nav sidebar)
    (BASE_URL, "/gnds/people/core-faculty", "/gnds/people/", "Faculty of Arts and Science"),
    # Geol handled by _harvest_geol (profiles at /geol/{name}, not /geol/people/)
    (BASE_URL, "/religion/people/faculty", "/religion/people/faculty/", "Faculty of Arts and Science"),
    (BASE_URL, "/sps/about/people", "/sps/aboutpeople/", "Faculty of Arts and Science"),
    # Sociology handled separately (paginated) — see PAGINATED_QUEENSU_DEPTS
]

# Departments that use /people-search pattern (profile links in server-rendered HTML)
ARTSCI_PEOPLE_SEARCH_DEPTS = [
    # (base_url, people_search_path, link_pattern, faculty)
    (BASE_URL, "/classics/people-search", "/classics/people-search/", "Faculty of Arts and Science"),
    (BASE_URL, "/filmandmedia/people-search", "/filmandmedia/people-search/", "Faculty of Arts and Science"),
    (BASE_URL, "/french/people-search", "/french/people-search/", "Faculty of Arts and Science"),
    (BASE_URL, "/employment-studies/people-search", "/employment-studies/people-search/", "Faculty of Arts and Science"),
    (BASE_URL, "/llcu/people-search", "/llcu/people-search/", "Faculty of Arts and Science"),
]

# Departments on subdomains: (base_url, people_page_path, link_prefix, faculty_name)
# people_page_path: the page to fetch (e.g. "/people/faculty")
# link_prefix: the URL prefix that profile links start with (e.g. "/people/faculty/")
#   If None, uses people_page_path + "/"
ARTSCI_SUBDOMAIN_DEPTS = [
    ("https://biology.queensu.ca", "/meet-the-department/people/faculty-members", "/meet-the-department/people/", "Faculty of Arts and Science"),
    ("https://www.econ.queensu.ca", "/people/faculty", None, "Faculty of Arts and Science"),
    ("https://www.chem.queensu.ca", "/people", None, "Faculty of Arts and Science"),
    ("https://www.cs.queensu.ca", "/people", None, "Faculty of Arts and Science"),
    ("https://sdm.queensu.ca", "/people", None, "Faculty of Arts and Science"),
    ("https://skhs.queensu.ca", "/people", None, "Faculty of Arts and Science"),
]


def _harvest_queensu_dept(dept_slug: str, faculty: str) -> list[tuple[str, str, str]]:
    """Scrape a queensu.ca/{dept}/people page for profile URLs.

    Uses HTML structure to distinguish profile links (inside views-row/teaser containers)
    from navigation/category links (inside menu-item containers).
    """
    people_url = f"{BASE_URL}/{dept_slug}/people"
    resp = _fetch(people_url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    seen = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if f"/{dept_slug}/people/" not in href:
            continue

        # Skip links inside nav menus (category pages)
        if link.find_parent("li", class_=re.compile(r"menu-item")):
            continue
        if link.find_parent("nav"):
            continue

        # Build full URL
        if href.startswith("http"):
            full_url = href
        else:
            full_url = f"{BASE_URL}{href}"
        clean = full_url.split("?")[0].rstrip("/")

        if clean == people_url or clean in seen:
            continue
        seen.add(clean)
        results.append((SCHOOL_NAME, faculty, clean))

    return results


def _harvest_subdomain_dept(base_url: str, people_path: str, faculty: str, link_prefix: str | None = None) -> list[tuple[str, str, str]]:
    """Scrape a subdomain department people page for profile URLs.

    Args:
        base_url: e.g. "https://www.econ.queensu.ca"
        people_path: e.g. "/people/faculty" — the page to fetch
        faculty: e.g. "Faculty of Arts and Science"
        link_prefix: URL prefix for profile links. If None, uses people_path + "/"
    """
    resp = _fetch(f"{base_url}{people_path}")
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    seen = set()

    # The prefix that profile URLs should start with
    profile_prefix = (link_prefix or people_path.rstrip("/") + "/")

    for link in soup.find_all("a", href=True):
        href = link["href"]

        # Only consider links whose path starts with the expected prefix
        # This filters out sibling pages like /people/directory when we want /people/faculty/*
        if not href.startswith(profile_prefix) and not href.startswith(base_url + profile_prefix):
            # Also check for /people/{Name} patterns (CS uses /people/First/Last)
            if people_path == "/people" and re.match(r"/people/[A-Z]", href):
                pass  # Allow these through
            else:
                continue

        # Skip nav menu links
        if link.find_parent("li", class_=re.compile(r"menu-item")):
            continue
        if link.find_parent("nav"):
            continue

        if href.startswith("http"):
            full_url = href
        elif href.startswith("/"):
            full_url = f"{base_url}{href}"
        else:
            full_url = f"{base_url}/{href}"
        clean = full_url.split("?")[0].rstrip("/")

        # Skip listing pages themselves
        if clean == f"{base_url}{people_path}".rstrip("/"):
            continue
        if clean not in seen:
            seen.add(clean)
            results.append((SCHOOL_NAME, faculty, clean))

    return results


# ──────────────────────────────────────────────────────────
# Paginated queensu.ca departments
# ──────────────────────────────────────────────────────────

# (base_url, people_path, link_pattern, faculty)
# These departments have pagination on their people pages.
PAGINATED_QUEENSU_DEPTS = [
    (BASE_URL, "/sociology/people", "/sociology/people-search/", "Faculty of Arts and Science"),
]


def _harvest_paginated_queensu_dept(base_url: str, people_path: str, link_pattern: str, faculty: str) -> list[tuple[str, str, str]]:
    """Scrape a paginated queensu.ca people page for profile URLs."""
    results = []
    seen = set()
    page = 0

    while True:
        url = f"{base_url}{people_path}" if page == 0 else f"{base_url}{people_path}?page={page}"
        resp = _fetch(url)
        if not resp:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        found_on_page = 0

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if not href.startswith(link_pattern):
                continue
            if link.find_parent("li", class_=re.compile(r"menu-item")):
                continue
            if link.find_parent("nav"):
                continue

            full_url = f"{base_url}{href}".split("?")[0].rstrip("/")
            if full_url not in seen:
                seen.add(full_url)
                results.append((SCHOOL_NAME, faculty, full_url))
                found_on_page += 1

        if found_on_page == 0 and page > 0:
            break
        page += 1
        polite_delay(1.5)
        if page > 20:
            break

    return results


# ──────────────────────────────────────────────────────────
# Fine Art (BFA) — profile links live in sidebar nav
# ──────────────────────────────────────────────────────────

def _harvest_bfa() -> list[tuple[str, str, str]]:
    """Scrape BFA faculty — links are in sidebar nav at /bfa/faculty/{name}."""
    faculty = "Faculty of Arts and Science"
    resp = _fetch(f"{BASE_URL}/bfa/faculty/faculty")
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    seen = set()
    skip = {"faculty", "careers", "staff"}

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/bfa/faculty/" not in href:
            continue
        slug = href.rstrip("/").split("/")[-1]
        if slug in skip:
            continue
        if href.startswith("http"):
            full_url = href.split("?")[0].rstrip("/")
        else:
            full_url = f"{BASE_URL}{href}".split("?")[0].rstrip("/")
        # Normalize to https
        full_url = full_url.replace("http://www.", "https://www.")
        if full_url not in seen:
            seen.add(full_url)
            results.append((SCHOOL_NAME, faculty, full_url))

    return results


# ──────────────────────────────────────────────────────────
# Geological Sciences (custom — profiles at /geol/{name})
# ──────────────────────────────────────────────────────────

def _harvest_geol() -> list[tuple[str, str, str]]:
    """Scrape geol faculty page — profiles use 'View Profile' links at /geol/{name}."""
    faculty = "Faculty of Arts and Science"
    resp = _fetch(f"{BASE_URL}/geol/faculty-research/faculty")
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    seen = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not href.startswith("/geol/"):
            continue
        # Skip nav and category pages
        if link.find_parent("li", class_=re.compile(r"menu-item")):
            continue
        if link.find_parent("nav"):
            continue
        # Skip non-profile pages (those with sub-paths like /faculty-research/)
        if "/faculty-research" in href:
            continue

        full_url = f"{BASE_URL}{href}".split("?")[0].rstrip("/")
        if full_url not in seen:
            seen.add(full_url)
            results.append((SCHOOL_NAME, faculty, full_url))

    return results


# ──────────────────────────────────────────────────────────
# Faculty of Education (paginated)
# ──────────────────────────────────────────────────────────

def _harvest_education() -> list[tuple[str, str, str]]:
    """Scrape educ.queensu.ca/people with pagination."""
    base = "https://educ.queensu.ca"
    faculty = "Faculty of Education"
    results = []
    seen = set()
    page = 0

    while True:
        url = f"{base}/people" if page == 0 else f"{base}/people?page={page}"
        resp = _fetch(url)
        if not resp:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        found_on_page = 0

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/people/" in href and href != "/people/":
                if href.startswith("http"):
                    full_url = href
                elif href.startswith("/"):
                    full_url = f"{base}{href}"
                else:
                    full_url = f"{base}/{href}"
                clean = full_url.split("?")[0].rstrip("/")
                if clean.endswith("/people"):
                    continue
                if clean not in seen:
                    seen.add(clean)
                    results.append((SCHOOL_NAME, faculty, clean))
                    found_on_page += 1

        # Stop if no new profiles found (end of pagination)
        if found_on_page == 0:
            break

        page += 1
        polite_delay(1.5)

        # Safety limit
        if page > 20:
            break

    return results


# ──────────────────────────────────────────────────────────
# Faculty of Law
# ──────────────────────────────────────────────────────────

def _harvest_law() -> list[tuple[str, str, str]]:
    """Scrape law.queensu.ca/directory for faculty profiles."""
    base = "https://law.queensu.ca"
    faculty = "Faculty of Law"
    resp = _fetch(f"{base}/directory")
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    seen = set()

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/directory/" in href and href != "/directory/":
            if href.startswith("http"):
                full_url = href
            elif href.startswith("/"):
                full_url = f"{base}{href}"
            else:
                full_url = f"{base}/{href}"
            clean = full_url.split("?")[0].rstrip("/")
            if clean.endswith("/directory"):
                continue
            if clean not in seen:
                seen.add(clean)
                results.append((SCHOOL_NAME, faculty, clean))

    return results


# ──────────────────────────────────────────────────────────
# Faculty of Health Sciences
# ──────────────────────────────────────────────────────────

def _harvest_nursing() -> list[tuple[str, str, str]]:
    """Scrape nursing.queensu.ca/faculty-staff for profiles."""
    base = "https://nursing.queensu.ca"
    faculty = "Faculty of Health Sciences"
    resp = _fetch(f"{base}/faculty-staff")
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    seen = set()

    # Nursing uses short profile URLs like /lastname
    # Look for links within the directory content area
    for link in soup.find_all("a", href=True):
        href = link["href"]
        # Nursing profiles are direct name slugs, but we need to filter carefully
        # They appear as links within person teaser cards
        parent = link.find_parent(class_=re.compile(r"person|profile|teaser|card|directory"))
        if parent and href.startswith("/") and not href.startswith("/faculty-staff"):
            full_url = f"{base}{href}".split("?")[0].rstrip("/")
            if full_url not in seen and full_url != base:
                seen.add(full_url)
                results.append((SCHOOL_NAME, faculty, full_url))

    return results


# ──────────────────────────────────────────────────────────
# Smith Engineering (JS-rendered — uses Playwright)
# ──────────────────────────────────────────────────────────

ENGINEERING_DEPTS = [
    ("chee", "Chemical Engineering"),
    ("civil", "Civil Engineering"),
    ("ece", "Electrical and Computer Engineering"),
    ("mme", "Mechanical and Materials Engineering"),
    ("mre", "Mechatronics and Robotics Engineering"),
    ("mining", "Mining Engineering"),
]


def _harvest_engineering_dept_playwright(dept_slug: str, dept_name: str) -> list[tuple[str, str, str]]:
    """Scrape a Smith Engineering department directory using Playwright."""
    faculty = "Smith Engineering"
    base = "https://smithengineering.queensu.ca"

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("    Playwright not available, skipping engineering departments")
        return []

    results = []
    seen = set()

    # Try common directory URL patterns
    dir_urls = [
        f"{base}/{dept_slug}/contact/directory.html",
        f"{base}/{dept_slug}/people",
        f"{base}/{dept_slug}/about/people",
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for dir_url in dir_urls:
            try:
                page.goto(dir_url, wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(3000)

                # Extract all links that look like profile pages
                links = page.eval_on_selector_all(
                    "a[href]",
                    "els => els.map(e => e.href)"
                )

                for link in links:
                    # Look for profile-like URLs
                    if any(pattern in link for pattern in ["/profile/", "/people/", "/directory/"]):
                        clean = link.split("?")[0].rstrip("/")
                        if clean not in seen:
                            seen.add(clean)
                            results.append((SCHOOL_NAME, faculty, clean))

                if results:
                    break  # Found profiles on this URL pattern

            except Exception as e:
                print(f"    Playwright error on {dir_url}: {e}")
                continue

        browser.close()

    return results


# Also handle queensu.ca-hosted engineering depts (Physics, Math, Geology)
# These are already in ARTSCI_QUEENSU_DEPTS since they're shared with Arts & Science


# ──────────────────────────────────────────────────────────
# Main entry points
# ──────────────────────────────────────────────────────────

def harvest_all() -> list[tuple[str, str, str]]:
    """Scrape all Queen's University departments for professor profile URLs."""
    all_urls = []

    # Arts & Science — queensu.ca departments
    for dept_slug, faculty in ARTSCI_QUEENSU_DEPTS:
        print(f"  Scraping {dept_slug} (queensu.ca)...")
        try:
            urls = _harvest_queensu_dept(dept_slug, faculty)
            all_urls.extend(urls)
            print(f"    Found {len(urls)} profiles")
        except Exception as e:
            print(f"    ERROR: {e}")
        polite_delay(2.0)

    # Arts & Science — departments with non-standard paths
    for base_url, people_path, link_prefix, faculty in ARTSCI_SPECIAL_DEPTS:
        dept_label = people_path.split("/")[1]  # e.g. "mathstat"
        print(f"  Scraping {dept_label} (special path)...")
        try:
            urls = _harvest_subdomain_dept(base_url, people_path, faculty, link_prefix)
            all_urls.extend(urls)
            print(f"    Found {len(urls)} profiles")
        except Exception as e:
            print(f"    ERROR: {e}")
        polite_delay(2.0)

    # Arts & Science — subdomain departments
    for base_url, people_path, link_prefix, faculty in ARTSCI_SUBDOMAIN_DEPTS:
        print(f"  Scraping {base_url}{people_path}...")
        try:
            urls = _harvest_subdomain_dept(base_url, people_path, faculty, link_prefix)
            all_urls.extend(urls)
            print(f"    Found {len(urls)} profiles")
        except Exception as e:
            print(f"    ERROR: {e}")
        polite_delay(2.0)

    # Arts & Science — departments using /people-search pattern
    for base_url, people_path, link_prefix, faculty in ARTSCI_PEOPLE_SEARCH_DEPTS:
        dept_label = people_path.split("/")[1]
        print(f"  Scraping {dept_label} (people-search)...")
        try:
            urls = _harvest_subdomain_dept(base_url, people_path, faculty, link_prefix)
            all_urls.extend(urls)
            print(f"    Found {len(urls)} profiles")
        except Exception as e:
            print(f"    ERROR: {e}")
        polite_delay(2.0)

    # Paginated queensu.ca departments
    for base_url, people_path, link_pattern, faculty in PAGINATED_QUEENSU_DEPTS:
        dept_label = people_path.split("/")[1]
        print(f"  Scraping {dept_label} (paginated)...")
        try:
            urls = _harvest_paginated_queensu_dept(base_url, people_path, link_pattern, faculty)
            all_urls.extend(urls)
            print(f"    Found {len(urls)} profiles")
        except Exception as e:
            print(f"    ERROR: {e}")
        polite_delay(2.0)

    # Fine Art (custom handler — nav sidebar)
    print("  Scraping bfa (custom)...")
    try:
        urls = _harvest_bfa()
        all_urls.extend(urls)
        print(f"    Found {len(urls)} profiles")
    except Exception as e:
        print(f"    ERROR: {e}")
    polite_delay(2.0)

    # Geological Sciences (custom handler)
    print("  Scraping geol (custom)...")
    try:
        urls = _harvest_geol()
        all_urls.extend(urls)
        print(f"    Found {len(urls)} profiles")
    except Exception as e:
        print(f"    ERROR: {e}")
    polite_delay(2.0)

    # Education
    print("  Scraping educ.queensu.ca...")
    try:
        urls = _harvest_education()
        all_urls.extend(urls)
        print(f"    Found {len(urls)} profiles")
    except Exception as e:
        print(f"    ERROR: {e}")
    polite_delay(2.0)

    # Law
    print("  Scraping law.queensu.ca...")
    try:
        urls = _harvest_law()
        all_urls.extend(urls)
        print(f"    Found {len(urls)} profiles")
    except Exception as e:
        print(f"    ERROR: {e}")
    polite_delay(2.0)

    # Nursing
    print("  Scraping nursing.queensu.ca...")
    try:
        urls = _harvest_nursing()
        all_urls.extend(urls)
        print(f"    Found {len(urls)} profiles")
    except Exception as e:
        print(f"    ERROR: {e}")
    polite_delay(2.0)

    # Smith Engineering (Playwright)
    for dept_slug, dept_name in ENGINEERING_DEPTS:
        print(f"  Scraping engineering/{dept_slug} (Playwright)...")
        try:
            urls = _harvest_engineering_dept_playwright(dept_slug, dept_name)
            all_urls.extend(urls)
            print(f"    Found {len(urls)} profiles")
        except Exception as e:
            print(f"    ERROR: {e}")
        polite_delay(2.0)

    return all_urls
