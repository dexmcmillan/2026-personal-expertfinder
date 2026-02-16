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
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB_PATH), help="SQLite database path"
    )
    parser.add_argument(
        "--output", type=str, default="embeddings.npy", help="Output numpy file"
    )
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
    else:
        print("All professors already have embeddings.")

    # Export all embeddings as numpy array
    all_profs = get_all_professors(db_path)
    all_with_emb = [p for p in all_profs if p["embedding"] is not None]
    if all_with_emb:
        matrix = np.array(
            [np.frombuffer(p["embedding"], dtype=np.float32) for p in all_with_emb]
        )
        np.save(args.output, matrix)
        print(
            f"Saved {matrix.shape[0]} embeddings ({matrix.shape[1]}d) to {args.output}"
        )
    else:
        print("No embeddings to export.")


if __name__ == "__main__":
    main()
