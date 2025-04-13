import numpy as np
from numba import cuda

@cuda.jit
def assign_clusters_batched(data, centroids, labels, n_clusters):
    batch_id = cuda.blockIdx.y
    sample_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if sample_id < data.shape[1]:
        min_dist = float('inf')
        label = -1
        for c in range(n_clusters):
            dist = 0.0
            for j in range(data.shape[2]):
                diff = data[batch_id, sample_id, j] - centroids[batch_id, c, j]
                dist += diff * diff
            if dist < min_dist:
                min_dist = dist
                label = c
        labels[batch_id, sample_id] = label

@cuda.jit
def update_centroids_batched(data, labels, centroids, counts):
    batch_id = cuda.blockIdx.y
    sample_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if sample_id < data.shape[1]:
        label = labels[batch_id, sample_id]
        for j in range(data.shape[2]):
            cuda.atomic.add(centroids, (batch_id, label, j), data[batch_id, sample_id, j])
        cuda.atomic.add(counts, (batch_id, label), 1)

def kmeans_cuda_batched(data_batch, n_clusters=3, max_iter=100, tol=1e-4, patience=3, threads_per_block=32):
    batch_size, n_samples, n_features = data_batch.shape
    data_batch = data_batch.astype(np.float32)

    centroids_init = np.empty((batch_size, n_clusters, n_features), dtype=np.float32)
    for i in range(batch_size):
        centroids_init[i] = data_batch[i][np.random.choice(n_samples, n_clusters, replace=False)]

    d_data = cuda.to_device(data_batch)
    d_centroids = cuda.to_device(centroids_init)
    d_labels = cuda.device_array((batch_size, n_samples), dtype=np.int32)

    blocks_per_grid_x = (n_samples + threads_per_block - 1) // threads_per_block
    blocks_per_grid = (blocks_per_grid_x, batch_size)

    no_improve_counts = np.zeros(batch_size, dtype=np.int32)
    prev_centroids = centroids_init.copy()

    for it in range(max_iter):
        assign_clusters_batched[blocks_per_grid, threads_per_block](d_data, d_centroids, d_labels, n_clusters)
        cuda.synchronize()

        new_centroids = np.zeros((batch_size, n_clusters, n_features), dtype=np.float32)
        counts = np.zeros((batch_size, n_clusters), dtype=np.int32)

        d_new_centroids = cuda.to_device(new_centroids)
        d_counts = cuda.to_device(counts)

        update_centroids_batched[blocks_per_grid, threads_per_block](d_data, d_labels, d_new_centroids, d_counts)
        cuda.synchronize()

        centroids_host = d_new_centroids.copy_to_host()
        counts_host = d_counts.copy_to_host()

        converged = 0
        for b in range(batch_size):
            for c in range(n_clusters):
                if counts_host[b, c] > 0:
                    centroids_host[b, c] /= counts_host[b, c]

            shift = np.linalg.norm(centroids_host[b] - prev_centroids[b])
            if shift < tol:
                no_improve_counts[b] += 1
            else:
                no_improve_counts[b] = 0

            if no_improve_counts[b] >= patience:
                converged += 1

        d_centroids = cuda.to_device(centroids_host)
        prev_centroids = centroids_host.copy()

        if converged == batch_size:
            break

    final_centroids = d_centroids.copy_to_host()
    final_labels = d_labels.copy_to_host()
    return final_centroids, final_labels

