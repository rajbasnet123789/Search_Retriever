"""
evaluate/run_evaluation.py

Run the 5 official evaluation queries against the indexed FAISS store
and output detailed results with score breakdowns.

Usage:
    python -m evaluate.run_evaluation
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retriever.search import FashionRetriever
from retriever.query_parser import parse_query

INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index_store"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))

# Official evaluation queries
EVAL_QUERIES = [
    {
        "id": "Q1",
        "category": "Attribute Specific",
        "query": "A person in a bright yellow raincoat.",
    },
    {
        "id": "Q2",
        "category": "Contextual/Place",
        "query": "Professional business attire inside a modern office.",
    },
    {
        "id": "Q3",
        "category": "Complex Semantic",
        "query": "Someone wearing a blue shirt sitting on a park bench.",
    },
    {
        "id": "Q4",
        "category": "Style Inference",
        "query": "Casual weekend outfit for a city walk.",
    },
    {
        "id": "Q5",
        "category": "Compositional",
        "query": "A red tie and a white shirt in a formal setting.",
    },
]

TOP_K = 10


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 80)
    print("GLANCE — EVALUATION SUITE")
    print("=" * 80)
    print(f"  Index directory : {INDEX_DIR}")
    print(f"  Results output  : {RESULTS_DIR}")
    print(f"  Top-K           : {TOP_K}")
    print()

    # Load retriever with pre-built index
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading retriever and FAISS index (device: {device})...")
    retriever = FashionRetriever(index_dir=INDEX_DIR, device=device, conf=0.25)

    global_count = retriever.store.count("global") if "global" in retriever.store.indices else 0
    regional_count = retriever.store.count("regional") if "regional" in retriever.store.indices else 0
    print(f"  Global vectors loaded  : {global_count}")
    print(f"  Regional vectors loaded: {regional_count}")
    print(f"  Metadata entries       : {len(retriever.metadata)}")
    print()

    if global_count == 0:
        print("ERROR: No indexed vectors found. Run 'python -m evaluate.index_all' first.")
        sys.exit(1)

    all_results = {}

    for eq in EVAL_QUERIES:
        qid = eq["id"]
        category = eq["category"]
        query = eq["query"]

        print("-" * 80)
        print(f"[{qid}] {category}")
        print(f"  Query: \"{query}\"")
        print()

        # Parse query
        parsed = parse_query(query)
        print(f"  Parsed garments : {json.dumps(parsed.get('garments', []), indent=4)}")
        print(f"  Parsed scene    : {json.dumps(parsed.get('scene', {}), indent=4)}")
        print()

        # Search
        t0 = time.time()
        results = retriever.search(query, k=TOP_K)
        elapsed = time.time() - t0

        print(f"  Search time: {elapsed:.3f}s")
        print(f"  Results ({len(results)}):")
        print()

        query_results = []
        for rank, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            scene_cat = meta.get("scene_category", "?")
            io = meta.get("indoor_outdoor", "?")
            garments = meta.get("garments", [])
            garment_labels = [f"{g['label']}({g.get('primary_color','?')})" for g in garments]

            print(f"    {rank:>2}. [{r['score']:.4f}]  {os.path.basename(r['path'])}")
            print(f"        global_clip={r['global_clip_score']:.3f} | "
                  f"regional_clip={r['regional_clip_score']:.3f} | "
                  f"comp={r['compositional_score']:.3f} | "
                  f"scene={r['scene_score']:.3f}")
            print(f"        Scene: {scene_cat} ({io}) | "
                  f"Garments: {', '.join(garment_labels) if garment_labels else 'None'}")
            print()

            query_results.append({
                "rank": rank,
                "path": r["path"],
                "filename": os.path.basename(r["path"]),
                "final_score": r["score"],
                "global_clip_score": r["global_clip_score"],
                "regional_clip_score": r["regional_clip_score"],
                "compositional_score": r["compositional_score"],
                "scene_score": r["scene_score"],
                "scene_category": scene_cat,
                "indoor_outdoor": io,
                "detected_garments": garment_labels,
            })

        all_results[qid] = {
            "query_id": qid,
            "category": category,
            "query": query,
            "parsed": parsed,
            "search_time_s": elapsed,
            "results": query_results,
        }

    # Save JSON results
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"  Full results saved to: {results_path}")
    print()

    # Summary table
    print(f"{'ID':<5} {'Category':<22} {'Top-1 Score':<13} {'Top-1 File':<45} {'Time':<8}")
    print("-" * 95)
    for eq in EVAL_QUERIES:
        qid = eq["id"]
        qr = all_results[qid]
        top1 = qr["results"][0] if qr["results"] else None
        if top1:
            print(f"{qid:<5} {eq['category']:<22} {top1['final_score']:<13.4f} "
                  f"{top1['filename']:<45} {qr['search_time_s']:.3f}s")
        else:
            print(f"{qid:<5} {eq['category']:<22} {'N/A':<13} {'N/A':<45} {qr['search_time_s']:.3f}s")


if __name__ == "__main__":
    main()
