import numpy as np
from numba import njit, prange

@njit(parallel=True)
def assign_clusters_batched_cpu(data_batch, centroids, labels):
    batch_size, n_samples, n_features = data_batch.shape
    n_clusters = centroids.shape[1]

    for b in prange(batch_size):
        for i in range(n_samples):
            min_dist = np.inf
            label = -1
            for c in range(n_clusters):
                dist = 0.0
                for j in range(n_features):
                    diff = data_batch[b, i, j] - centroids[b, c, j]
                    dist += diff * diff
                if dist < min_dist:
                    min_dist = dist
                    label = c
            labels[b, i] = label

@njit(parallel=True)
def update_centroids_batched_cpu(data_batch, labels, centroids_out, counts_out):
    batch_size, n_samples, n_features = data_batch.shape

    for b in prange(batch_size):
        for i in range(n_samples):
            label = labels[b, i]
            for j in range(n_features):
                centroids_out[b, label, j] += data_batch[b, i, j]
            counts_out[b, label] += 1

@njit(parallel=True)
def finalize_centroids_batched_cpu(centroids, counts):
    batch_size, n_clusters, n_features = centroids.shape

    for b in prange(batch_size):
        for c in range(n_clusters):
            if counts[b, c] > 0:
                for j in range(n_features):
                    centroids[b, c, j] /= counts[b, c]

@njit(parallel=True)
def compute_centroid_shifts(prev_centroids, new_centroids, shifts):
    batch_size = prev_centroids.shape[0]

    for b in prange(batch_size):
        diff = 0.0
        for c in range(prev_centroids.shape[1]):
            for j in range(prev_centroids.shape[2]):
                d = new_centroids[b, c, j] - prev_centroids[b, c, j]
                diff += d * d
        shifts[b] = np.sqrt(diff)

def kmeans_cpu_batched(data_batch, n_clusters=3, n_iter=100, tol=1e-4, patience=3):
    batch_size, n_samples, n_features = data_batch.shape
    data_batch = data_batch.astype(np.float32)

    centroids = np.empty((batch_size, n_clusters, n_features), dtype=np.float32)
    for b in range(batch_size):
        centroids[b] = data_batch[b][np.random.choice(n_samples, n_clusters, replace=False)]

    labels = np.empty((batch_size, n_samples), dtype=np.int32)
    no_improve_counts = np.zeros(batch_size, dtype=np.int32)
    prev_centroids = centroids.copy()

    for it in range(n_iter):
        assign_clusters_batched_cpu(data_batch, centroids, labels)

        new_centroids = np.zeros_like(centroids)
        counts = np.zeros((batch_size, n_clusters), dtype=np.int32)

        update_centroids_batched_cpu(data_batch, labels, new_centroids, counts)
        finalize_centroids_batched_cpu(new_centroids, counts)

        shifts = np.zeros(batch_size, dtype=np.float32)
        compute_centroid_shifts(prev_centroids, new_centroids, shifts)

        converged = 0
        for b in range(batch_size):
            if shifts[b] < tol:
                no_improve_counts[b] += 1
            else:
                no_improve_counts[b] = 0

            if no_improve_counts[b] >= patience:
                converged += 1

        prev_centroids = new_centroids.copy()
        centroids = new_centroids

        if converged == batch_size:
            break

    return centroids, labels
