import numpy as np

def dbscan(X, eps, min_samples):
    labels = np.full(X.shape[0], -1) 
    cluster_id = 0

    def region_query(point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        neighbors = np.where(distances <= eps)[0]
        return neighbors

    def expand_cluster(point_idx, neighbors):
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1: 
                labels[neighbor_idx] = cluster_id
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))
            i += 1

    for i in range(X.shape[0]):
        if labels[i] != -1: 
            continue
        neighbors = region_query(i)
        if len(neighbors) < min_samples:
            labels[i] = -1  
        else:
            expand_cluster(i, neighbors)
            cluster_id += 1  

    return labels