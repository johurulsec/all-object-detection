import numpy as np
from sklearn.cluster import AgglomerativeClustering

CANNOT_LINK = 2.0  # larger than any cosine distance


def distance_matrix(embeddings, spans):
    """Cosine distance between L2-normalised track embeddings.

    Tracks that overlap in time cannot be the same person, so their distance is set to CANNOT_LINK.
    """
    x = np.stack(embeddings)
    d = 1.0 - x @ x.T
    for i, (a0, a1) in enumerate(spans):
        for j, (b0, b1) in enumerate(spans):
            if i != j and a0 <= b1 and b0 <= a1:
                d[i, j] = CANNOT_LINK
    np.fill_diagonal(d, 0.0)
    return d


def assign_person_ids(embeddings, spans, threshold=0.4):
    """Cluster tracks into people. Returns a list of cluster ints, one per track."""
    if len(embeddings) < 2:
        return [0] * len(embeddings)
    d = distance_matrix(embeddings, spans)
    clustering = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average", distance_threshold=threshold)
    return clustering.fit_predict(d).tolist()
