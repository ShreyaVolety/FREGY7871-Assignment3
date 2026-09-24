"""Deduplicate news, cluster event coverage, and generate article-level features."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .common import ROOT, ensure_dirs, load_config
from .news_utils import keyword_scores, normalize_text, stable_id


class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return int(root)

    def union(self, left, right):
        a, b = self.find(int(left)), self.find(int(right))
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1

    def labels(self):
        roots = np.array([self.find(i) for i in range(len(self.parent))])
        _, labels = np.unique(roots, return_inverse=True)
        return labels


def event_clusters(frame, threshold=0.52, hours=48, progress_every=10):
    """Cluster similar articles without constructing an N x N dense matrix.

    TF-IDF is built once. Cosine similarities are computed only for pairs of
    calendar-day blocks that could satisfy the configured timestamp window.
    Qualifying pairs are joined with union-find.
    """
    text = (
        frame["headline"].fillna("") + " " + frame["article_text"].fillna("")
    ).map(normalize_text)
    print(f"Building sparse TF-IDF matrix for {len(frame):,} articles...", flush=True)
    matrix = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.98,
        max_features=150_000, stop_words="english", sublinear_tf=True,
        dtype=np.float32,
    ).fit_transform(text)
    print(f"TF-IDF shape: {matrix.shape[0]:,} x {matrix.shape[1]:,}; "
          f"nonzero values: {matrix.nnz:,}", flush=True)

    times = pd.to_datetime(frame["published_at_utc"], utc=True)
    time_ns = times.astype("int64").to_numpy()
    day_values = times.dt.floor("D")
    unique_days = pd.Index(day_values.unique()).sort_values()
    groups = {day: np.flatnonzero(day_values.eq(day).to_numpy()) for day in unique_days}
    union_find = UnionFind(len(frame))
    max_day_gap = int(np.ceil(hours / 24))
    maximum_ns = hours * 3_600_000_000_000
    comparisons, links = 0, 0

    for day_number, day in enumerate(unique_days):
        left_indices = groups[day]
        for offset in range(max_day_gap + 1):
            other_day = day + pd.Timedelta(days=offset)
            if other_day not in groups:
                continue
            right_indices = groups[other_day]
            similarities = cosine_similarity(
                matrix[left_indices], matrix[right_indices], dense_output=False
            ).tocoo()
            candidate_mask = similarities.data >= threshold
            rows = similarities.row[candidate_mask]
            cols = similarities.col[candidate_mask]
            for local_left, local_right in zip(rows, cols):
                global_left = left_indices[local_left]
                global_right = right_indices[local_right]
                if global_left >= global_right:
                    continue
                comparisons += 1
                if abs(time_ns[global_left] - time_ns[global_right]) <= maximum_ns:
                    union_find.union(global_left, global_right)
                    links += 1
        completed = day_number + 1
        if completed % progress_every == 0 or completed == len(unique_days):
            print(f"Cluster progress: {completed}/{len(unique_days)} days "
                  f"({100*completed/len(unique_days):.1f}%); "
                  f"{links:,} similarity links", flush=True)
    return union_find.labels()


def main():
    ensure_dirs()
    cfg = load_config()["news"]
    frame = pd.read_csv(ROOT / "data/raw/articles_raw.csv")
    if frame.empty:
        raise SystemExit("No articles found. Inspect outputs/news_collection_failures.csv.")
    frame["headline"] = frame["headline"].fillna("")
    frame["article_text"] = frame["article_text"].fillna("")
    frame["normalized_headline"] = frame["headline"].map(normalize_text)
    # Exact URL and exact normalized-title duplicates are removed first.
    frame = frame.sort_values("published_at_utc").drop_duplicates("url")
    nonempty = frame["normalized_headline"].ne("")
    frame = pd.concat([
        frame.loc[nonempty].drop_duplicates(["normalized_headline", "publication"]),
        frame.loc[~nonempty],
    ]).sort_values("published_at_utc")
    frame = frame.reset_index(drop=True)
    frame["cluster_number"] = event_clusters(
        frame, float(cfg["cluster_similarity"]), int(cfg["cluster_hours"]))
    cluster_keys = {
        number: stable_id(group["published_at_utc"].iloc[0], group["headline"].iloc[0])
        for number, group in frame.groupby("cluster_number", sort=False)
    }
    frame["event_cluster_id"] = frame["cluster_number"].map(cluster_keys)
    scored = frame.apply(lambda r: keyword_scores(f"{r.headline} {r.article_text}"), axis=1)
    frame[["direction_auto", "escalation_score", "deescalation_score", "market_keyword_score"]] = \
        pd.DataFrame(scored.tolist(), index=frame.index)
    frame["market_relevance_score"] = np.maximum(frame["market_keyword_score"],
                                                  0.5 * frame["financial_source"])
    cluster = frame.groupby("event_cluster_id").agg(
        first_published_at_utc=("published_at_utc", "min"),
        last_published_at_utc=("published_at_utc", "max"),
        representative_headline=("headline", "first"),
        article_count=("article_id", "size"),
        publication_count=("publication", "nunique"),
        source_group_count=("source_group", "nunique"),
        market_relevance_score=("market_relevance_score", "max"),
    ).reset_index()
    frame.to_csv(ROOT / "data/processed/articles_deduplicated.csv", index=False)
    cluster.to_csv(ROOT / "data/processed/event_clusters.csv", index=False)
    print(f"Retained {len(frame):,} articles in {len(cluster):,} event clusters")


if __name__ == "__main__":
    main()
