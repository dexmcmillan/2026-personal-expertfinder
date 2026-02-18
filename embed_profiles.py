"""
Generate embeddings for professor expertise text using SPECTER2.

SPECTER2 (Allen Institute for AI) is trained on millions of scientific paper
abstracts via citation graphs, making it significantly better than general
sentence models at representing academic disciplines and their relationships.

Usage:
    uv run python embed_profiles.py           # embed all professors without embeddings
    uv run python embed_profiles.py --reembed # clear and re-embed everything
    uv run python embed_profiles.py --db professors.db
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from db import (
    clear_embeddings,
    get_all_professors,
    get_professors_without_embeddings,
    update_embedding,
    DEFAULT_DB_PATH,
)

BASE_MODEL = "allenai/specter2_base"
ADAPTER   = "allenai/specter2"        # proximity adapter: best for clustering


def load_model():
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    print(f"Loading {BASE_MODEL} + proximity adapter...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoAdapterModel.from_pretrained(BASE_MODEL)
    model.load_adapter(ADAPTER, source="hf", load_as="specter2", set_active=True)
    model.eval()
    print("  Model ready.")
    return tokenizer, model


def embed_texts(texts: list[str], batch_size: int = 16) -> np.ndarray:
    import torch

    tokenizer, model = load_model()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token is the document embedding in SPECTER2
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate SPECTER2 expertise embeddings")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--output", default="embeddings.npy")
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Clear all existing embeddings and re-embed from scratch",
    )
    args = parser.parse_args()

    db_path = Path(args.db)

    if args.reembed:
        cleared = clear_embeddings(db_path)
        print(f"Cleared {cleared:,} existing embeddings.")

    profs = get_professors_without_embeddings(db_path)
    if not profs:
        print("All professors already have embeddings.")
    else:
        print(f"Embedding {len(profs):,} professors...")
        texts = [p["expertise_raw"] or "" for p in profs]
        embeddings = embed_texts(texts)
        for prof, emb in zip(profs, embeddings):
            update_embedding(db_path, prof["profile_url"], emb)
        print(f"Saved {len(profs):,} embeddings to DB.")

    # Export full matrix for reference
    all_profs = get_all_professors(db_path)
    all_with_emb = [p for p in all_profs if p["embedding"] is not None]
    if all_with_emb:
        matrix = np.array(
            [np.frombuffer(p["embedding"], dtype=np.float32) for p in all_with_emb]
        )
        np.save(args.output, matrix)
        print(f"Exported {matrix.shape[0]:,} embeddings ({matrix.shape[1]}d) → {args.output}")


if __name__ == "__main__":
    main()
