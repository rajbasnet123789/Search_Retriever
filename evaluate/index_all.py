"""
evaluate/index_all.py

Index ALL images from D:\\val_test2020\\test into the FAISS store.
This builds both the 'global' and 'regional' indices plus metadata.json.

Usage:
    python -m evaluate.index_all
"""
import glob
import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retriever.search import FashionRetriever

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "index_store")
INDEX_DIR = os.path.abspath(INDEX_DIR)

IMG_DIR = r"D:\val_test2020\test"

CONF = 0.25


def main():
    print("=" * 70)
    print("GLANCE — FULL DATASET INDEXING")
    print("=" * 70)
    print(f"  Image directory : {IMG_DIR}")
    print(f"  Index output    : {INDEX_DIR}")
    print()
    # Auto-detect GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device          : {device}")
    print()

    # Collect all jpg images
    all_images = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if not all_images:
        print(f"ERROR: No .jpg images found in {IMG_DIR}")
        sys.exit(1)

    total = len(all_images)
    print(f"  Total images found: {total}")
    print()

    # Initialize retriever (loads existing index if present)
    retriever = FashionRetriever(index_dir=INDEX_DIR, device=device, conf=CONF)

    # Check if we already have an index with the same count
    existing_global = retriever.store.count("global") if "global" in retriever.store.indices else 0
    if existing_global >= total:
        print(f"  Index already contains {existing_global} global vectors (>= {total} images).")
        print("  Skipping indexing. Delete index_store/ to re-index.")
        return

    t_start = time.time()
    success = 0
    failed = 0

    for i, img_path in enumerate(all_images):
        try:
            retriever.index_image(img_path)
            success += 1
        except Exception as e:
            failed += 1
            print(f"  SKIP [{i+1}/{total}] {os.path.basename(img_path)}: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1:>5}/{total}] indexed | {success} ok, {failed} failed | "
                  f"{elapsed:.1f}s elapsed ({rate:.2f} img/s)")

    # Save
    retriever.save(INDEX_DIR)

    t_total = time.time() - t_start
    print()
    print("=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)
    print(f"  Images indexed  : {success} / {total}  ({failed} failed)")
    print(f"  Global vectors  : {retriever.store.count('global')}")
    print(f"  Regional vectors: {retriever.store.count('regional')}")
    print(f"  Total time      : {t_total:.1f}s")
    print(f"  Index saved to  : {INDEX_DIR}")
    print()


if __name__ == "__main__":
    main()
