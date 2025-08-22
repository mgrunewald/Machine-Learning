import numpy as np

# paso 1: inicializar aleatoriamente los centroides
def initialize_centroids(X, K, N):
    return X[np.random.choice(N, K, replace=False)]

# paso 2: crear la matriz R con OHE para los clusters a los que pertenece la muestra (para despues minimizar sobre esto)
def assign_clusters(X, centroids, N):
    K = centroids.shape[0]
    r = np.zeros((N, K))
    for i, point in enumerate(X):
        distances = [np.sum((point - centroid) ** 2) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        r[i, closest_centroid] = 1
    return r

# paso 3: minimizar el error con respecto a los centroides
def update_centroids(X, r, K, N):
    new_centroids = []
    for j in range(K):
        points_in_cluster = X[r[:, j] == 1]
        if len(points_in_cluster) > 0:
            new_centroids.append(np.mean(points_in_cluster, axis=0))
        else:
            new_centroids.append(X[np.random.choice(N)])
    return np.array(new_centroids)

def calculate_L(X, r, centroids):
    L = 0
    for j in range(centroids.shape[0]):
        points_in_cluster = X[r[:, j] == 1]
        L += np.sum(np.sum((points_in_cluster - centroids[j]) ** 2, axis=1))
    return L

# algoritmo
def kmeans(X, K, max_iters=100, tol=1e-4):
    N = X.shape[0]
    centroids = initialize_centroids(X, K, N)
    for _ in range(max_iters):
        r = assign_clusters(X, centroids, N)
        new_centroids = update_centroids(X, r, K, N)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    L = calculate_L(X, r, centroids)
    return r, centroids, L