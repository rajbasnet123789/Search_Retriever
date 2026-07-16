"""
evaluate/run_evaluation.py

Run the 5 official evaluation queries against the indexed FAISS store
and output detailed results with score breakdowns and quantitative metrics.

Usage:
    python -m evaluate.run_evaluation
"""
import csv
import json
import math
import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retriever.search import FashionRetriever
from retriever.query_parser import parse_query

INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index_store"))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
PARSED_CACHE_PATH = os.path.join(RESULTS_DIR, "parsed_queries.json")

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

RELEVANCE = {}


def precision_at_k(relevances, k):
    top = relevances[:k]
    return sum(1 for r in top if r >= 1) / k if k > 0 else 0.0


def mrr(relevances):
    for i, r in enumerate(relevances, 1):
        if r >= 1:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevances, k):
    def dcg(scores):
        return sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores[:k]))
    actual_dcg = dcg(relevances)
    ideal = sorted(relevances, reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def load_parsed_cache():
    if os.path.exists(PARSED_CACHE_PATH):
        with open(PARSED_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_parsed_cache(cache):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PARSED_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def get_parsed_query(query, cache):
    if query in cache:
        return cache[query]
    parsed = parse_query(query)
    cache[query] = parsed
    return parsed


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 80)
    print("GLANCE — EVALUATION SUITE")
    print("=" * 80)
    print(f"  Index directory : {INDEX_DIR}")
    print(f"  Results output  : {RESULTS_DIR}")
    print(f"  Top-K           : {TOP_K}")
    print()

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

    parsed_cache = load_parsed_cache()
    all_results = {}

    for eq in EVAL_QUERIES:
        qid = eq["id"]
        category = eq["category"]
        query = eq["query"]

        print("-" * 80)
        print(f"[{qid}] {category}")
        print(f"  Query: \"{query}\"")
        print()

        parsed = get_parsed_query(query, parsed_cache)
        source = parsed.get("parser_source", "unknown")
        reason = parsed.get("fallback_reason", "")
        print(f"  Parser: {source}" + (f" ({reason})" if reason else ""))
        print(f"  Parsed garments : {json.dumps(parsed.get('garments', []), indent=4)}")
        print(f"  Parsed scene    : {json.dumps(parsed.get('scene', {}), indent=4)}")
        style_terms = parsed.get("style_terms", [])
        if style_terms:
            print(f"  Style terms     : {json.dumps(style_terms)}")
        print()

        t0 = time.time()
        results = retriever.search(query, k=TOP_K, parsed_query=parsed)
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

    save_parsed_cache(parsed_cache)

    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "image_path", "rank", "relevance"])
        for qid, qr in all_results.items():
            for r in qr["results"]:
                writer.writerow([qid, r["path"], r["rank"], ""])

    print("=" * 80)
    print("QUANTITATIVE METRICS")
    print("=" * 80)

    has_relevance = bool(RELEVANCE)
    if has_relevance:
        all_p1, all_p5, all_mrr, all_ndcg = [], [], [], []
        for eq in EVAL_QUERIES:
            qid = eq["id"]
            qr = all_results[qid]
            filenames = [r["filename"] for r in qr["results"]]
            relevances = [RELEVANCE.get(qid, {}).get(f, 0) for f in filenames]
            p1 = precision_at_k(relevances, 1)
            p5 = precision_at_k(relevances, 5)
            rr = mrr(relevances)
            ndcg = ndcg_at_k(relevances, 5)
            all_p1.append(p1)
            all_p5.append(p5)
            all_mrr.append(rr)
            all_ndcg.append(ndcg)
            print(f"  {qid}: P@1={p1:.3f}  P@5={p5:.3f}  MRR={rr:.3f}  NDCG@5={ndcg:.3f}")
        print()
        print(f"  Macro-average: P@1={sum(all_p1)/len(all_p1):.3f}  "
              f"P@5={sum(all_p5)/len(all_p5):.3f}  "
              f"MRR={sum(all_mrr)/len(all_mrr):.3f}  "
              f"NDCG@5={sum(all_ndcg)/len(all_ndcg):.3f}")
    else:
        print("  No relevance judgments found.")
        print("  Annotate results in evaluation_results.csv (set relevance: 2=full, 1=partial, 0=incorrect)")
        print("  Then copy relevant rows into RELEVANCE dict in this script and re-run.")

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"  JSON results: {results_path}")
    print(f"  CSV for annotation: {csv_path}")
    print(f"  Parsed query cache: {PARSED_CACHE_PATH}")
    print()

    print(f"{'ID':<5} {'Category':<22} {'Results':<9} {'Top-1 Score':<13} {'Top-1 File':<45} {'Time':<8}")
    print("-" * 102)
    for eq in EVAL_QUERIES:
        qid = eq["id"]
        qr = all_results[qid]
        top1 = qr["results"][0] if qr["results"] else None
        n_results = len(qr["results"])
        if top1:
            print(f"{qid:<5} {eq['category']:<22} {n_results:<9} {top1['final_score']:<13.4f} "
                  f"{top1['filename']:<45} {qr['search_time_s']:.3f}s")
        else:
            print(f"{qid:<5} {eq['category']:<22} {n_results:<9} {'N/A':<13} {'N/A':<45} {qr['search_time_s']:.3f}s")


if __name__ == "__main__":
    main()
