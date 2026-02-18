"""
Remove non-academic staff and students from the professors database.

Uses a keep-list: records are kept only if their title is NULL (unknown)
or contains a recognized academic keyword. Everything else is deleted.

Usage:
    uv run python cleanup_non_academics.py            # dry run (preview only)
    uv run python cleanup_non_academics.py --execute  # actually delete
"""

import argparse
from collections import Counter
from pathlib import Path
import sqlite3

from db import DEFAULT_DB_PATH

# If a title contains any of these (case-insensitive), the record is kept.
# NULL titles are always kept — we can't tell from the page alone.
ACADEMIC_KEYWORDS = [
    "professor",
    "instructor",
    "adjunct",
    "lecturer",
    "fellow",           # teaching fellow, post-doctoral fellow
    "emeritus",
    "emeriti",
    "faculty",          # adjunct faculty, etc.
    "scientist",
    "scholar",
    "research associate",
    "head",             # department head (almost always a professor)
    "chair",            # department/program chair (almost always a professor)
    "director",         # research/clinic/school directors (usually academic roles)
]


def is_academic(title: str | None) -> bool:
    if title is None:
        return True
    t = title.lower()
    return any(kw in t for kw in ACADEMIC_KEYWORDS)


def main():
    parser = argparse.ArgumentParser(description="Remove non-academic records from professors.db")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete records (default is dry run)")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id, name, title, school, department FROM professors"
    ).fetchall()

    to_keep = [r for r in rows if is_academic(r["title"])]
    to_delete = [r for r in rows if not is_academic(r["title"])]

    print(f"Total records:    {len(rows):,}")
    print(f"Keeping:          {len(to_keep):,}")
    print(f"Deleting:         {len(to_delete):,}")
    print()

    # Breakdown by school
    school_counts = Counter(r["school"] for r in to_delete)
    print("Deletions by school:")
    for school, count in school_counts.most_common():
        print(f"  {count:4d}  {school}")
    print()

    # Breakdown by title
    title_counts = Counter(r["title"] for r in to_delete)
    print("Titles being removed:")
    for title, count in title_counts.most_common(30):
        print(f"  {count:3d}  {title}")
    if len(title_counts) > 30:
        print(f"  ... and {len(title_counts) - 30} more title types")
    print()

    if not args.execute:
        print("DRY RUN — no changes made. Re-run with --execute to delete.")
        conn.close()
        return

    ids = [r["id"] for r in to_delete]
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM professors WHERE id IN ({placeholders})", ids)
    conn.commit()

    remaining = conn.execute("SELECT COUNT(*) FROM professors").fetchone()[0]
    print(f"Deleted {len(to_delete):,} records. {remaining:,} remain.")
    conn.close()


if __name__ == "__main__":
    main()
