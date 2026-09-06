from pathlib import Path
from typing import Optional, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

base_dir = Path(__file__).resolve().parent

embedding_candidates = [
    base_dir / "receipts_embeddings_400k.npz",
    base_dir / "recetas_embeddings_400k.npz",
]

index_path = base_dir / "recipes_faiss.index"
ids_path = base_dir / "recipes_faiss_ids.npy"


def find_embeddings_file() -> Path:
    for candidate in embedding_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"The .npz file was not found. Found: {[str(p) for p in embedding_candidates]}"
    )


def load_embeddings(npz_path: Path):
    data = np.load(npz_path)
    if "ids" not in data or "embeddings" not in data:
        raise KeyError("The .npz file must contain 'ids' and 'embeddings' keys.")

    ids = data["ids"].astype(np.int64)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must have 2 dimensions, received: {embeddings.shape}"
        )

    if ids.shape[0] != embeddings.shape[0]:
        raise ValueError(
            "The number of ids does not match the embeddings quantity: "
            f"ids={ids.shape[0]}, embeddings={embeddings.shape[0]}"
        )

    return ids, embeddings


def build_faiss_index(embeddings: np.ndarray, normalize: bool = True):
    dim = embeddings.shape[1]

    if normalize:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    return index


def save_index(index, ids: np.ndarray):
    faiss.write_index(index, str(index_path))
    np.save(ids_path, ids)
    print(f"Index FAISS guardado en: {index_path}")
    print(f"IDs guardados en: {ids_path}")


def load_or_build_faiss_index():
    """Load a FAISS index of buid one if it does not exists."""
    if index_path.exists() and ids_path.exists():
        print(f"Loading FAISS index from: {index_path}")
        index = faiss.read_index(str(index_path))
        ids = np.load(ids_path)
        return index, ids.astype(np.int64)

    print("No FAISS index was found. Generating a new one...")
    npz_file = find_embeddings_file()
    print(f"Loading embeddings file from: {npz_file}")
    ids, embeddings = load_embeddings(npz_file)

    print(f"Number of registers: {len(ids)}")
    print(f"Embeddings dimension: {embeddings.shape[1]}")

    index = build_faiss_index(embeddings, normalize=True)
    save_index(index, ids)
    return index, ids


def search_similar_recipes(
    query: Union[str, np.ndarray],
    *,
    model: Optional[SentenceTransformer] = None,
    top_k: int = 10,
    index=None,
    ids=None,
):
    """Search recipes which are similar to a given input vector or text"""
    if index is None or ids is None:
        index, ids = load_or_build_faiss_index()

    if isinstance(query, str):
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
    else:
        query_embedding = np.asarray(query, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.shape[1] != index.d:
            raise ValueError(
                f"Input vector dimension ({query_embedding.shape[1]}) "
                f"does not match the index dimension ({index.d})."
            )
        faiss.normalize_L2(query_embedding)

    k = min(top_k, len(ids))
    distances, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({"id": int(ids[idx]), "score": float(score)})

    return results


def main():
    index, ids = load_or_build_faiss_index()
    print(f"Index ready. Total recipes: {len(ids)}")


if __name__ == "__main__":
    main()
