import glob
import os
import random
import time

from retriever.search import FashionRetriever

INDEX_DIR = r"D:\Glance\index_store"
IMG_DIR = r"D:\val_test2020\test"
NUM_IMAGES = 20  # Sample subset for rapid evaluation/verification

retriever = FashionRetriever(index_dir=INDEX_DIR, device="cpu", conf=0.25)

all_images = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
random.seed(42)
if len(all_images) > NUM_IMAGES:
    all_images = random.sample(all_images, NUM_IMAGES)

print(f"Found and sampled {len(all_images)} images from {IMG_DIR} to index")

for i, img in enumerate(all_images):
    try:
        retriever.index_image(img)
        if (i + 1) % 5 == 0 or (i + 1) == len(all_images):
            print(f"  Indexed {i + 1}/{len(all_images)}")
    except Exception as e:
        print(f"  Skip {os.path.basename(img)}: {e}")

retriever.save(INDEX_DIR)
print(f"\nIndex saved to {INDEX_DIR}")
print(f"  global vectors: {retriever.store.count('global')}")
print(f"  regional vectors: {retriever.store.count('regional')}")

queries = [
    "a yellow raincoat and black pants in a rainy street",
    "a blue shirt and a brown belt in a formal office setting",
    "a person in a bright red dress walking on the city sidewalk",
    "casual weekend streetwear with white sneakers in a park",
    "formal business blazer and tie inside a modern workplace desk area",
]

print("\n" + "=" * 70)
print("COMPOSITIONAL & REGIONAL EVALUATION QUERIES")
print("=" * 70)

for q in queries:
    t0 = time.time()
    results = retriever.search(q, k=3)
    elapsed = time.time() - t0
    print(f"\n>>> {q}  ({elapsed:.2f}s)")
    for i, r in enumerate(results):
        meta = r.get("metadata", {})
        scene_cat = meta.get("scene_category", "unknown")
        detected_garments = [g["label"] for g in meta.get("garments", [])]
        print(f"  {i + 1}. [{r['score']:.4f}] {os.path.basename(r['path'])}")
        print(f"     global_clip={r['global_clip_score']:.3f} | regional_clip={r['regional_clip_score']:.3f} | comp={r['compositional_score']:.3f} | scene={r['scene_score']:.3f}")
        print(f"     [Scene: {scene_cat}] | [Garments Detected: {', '.join(detected_garments) if detected_garments else 'None'}]")
