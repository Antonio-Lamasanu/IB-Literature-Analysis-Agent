"""
Regenerate embeddings for all documents that have chunks but no .npy file.

Run once after deploying the hybrid-retrieval upgrade to backfill existing documents:

    cd backend
    python scripts/regenerate_embeddings.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from embeddings import encode_texts, embeddings_path_for_chunks

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
REGISTRY_PATH = OUTPUTS_DIR / "documents.index.json"


def main() -> None:
    if not REGISTRY_PATH.exists():
        print(f"Registry not found at {REGISTRY_PATH}")
        sys.exit(1)

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    docs = registry.get("documents", [])
    print(f"Found {len(docs)} document(s) in registry.")

    for doc in docs:
        filename = doc.get("filename", "<unknown>")

        if not doc.get("chunks_available") or not doc.get("chunks_path"):
            print(f"  SKIP  {filename}: no chunks")
            continue

        chunks_path = Path(doc["chunks_path"])
        if not chunks_path.exists():
            print(f"  SKIP  {filename}: chunks file missing at {chunks_path}")
            continue

        emb_path = embeddings_path_for_chunks(chunks_path)
        if emb_path.exists():
            print(f"  OK    {filename}: embeddings already exist ({emb_path.name})")
            continue

        print(f"  GEN   {filename} ...", end=" ", flush=True)
        texts: list[str] = []
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                text = str(chunk.get("content", {}).get("text") or "").strip()
                texts.append(text)

        if not texts:
            print("WARNING: no text found in chunks file")
            continue

        emb = encode_texts(texts)
        if emb.shape[0] != len(texts):
            print(f"ERROR: shape mismatch ({emb.shape[0]} vs {len(texts)} chunks)")
            continue

        np.save(str(emb_path), emb)
        print(f"saved {emb.shape[0]} embeddings → {emb_path.name}")


if __name__ == "__main__":
    main()
