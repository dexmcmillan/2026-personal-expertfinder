import json
import sqlite3

import numpy as np
import pytest

from db import (
    init_db,
    insert_professor,
    get_professor_by_url,
    get_all_professors,
    get_professors_without_embeddings,
    update_embedding,
)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _make_prof(**overrides):
    defaults = {
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
    defaults.update(overrides)
    return defaults


def test_init_db_creates_table(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='professors'"
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_insert_and_retrieve_professor(db_path):
    prof = _make_prof()
    insert_professor(db_path, prof)
    result = get_professor_by_url(db_path, prof["profile_url"])
    assert result is not None
    assert result["name"] == "Jane Smith"
    assert result["school"] == "University of Waterloo"
    assert json.loads(result["expertise_keywords"]) == [
        "machine learning",
        "computer vision",
        "neural networks",
    ]


def test_insert_duplicate_url_skips(db_path):
    prof = _make_prof()
    insert_professor(db_path, prof)
    insert_professor(db_path, prof)
    all_profs = get_all_professors(db_path)
    assert len(all_profs) == 1


def test_get_professors_without_embeddings(db_path):
    insert_professor(db_path, _make_prof())
    without = get_professors_without_embeddings(db_path)
    assert len(without) == 1
    assert without[0]["name"] == "Jane Smith"


def test_update_embedding(db_path):
    prof = _make_prof()
    insert_professor(db_path, prof)
    embedding = np.random.rand(384).astype(np.float32)
    update_embedding(db_path, prof["profile_url"], embedding)
    result = get_professor_by_url(db_path, prof["profile_url"])
    assert result["embedding"] is not None
    restored = np.frombuffer(result["embedding"], dtype=np.float32)
    assert restored.shape == (384,)
    assert np.allclose(restored, embedding)
