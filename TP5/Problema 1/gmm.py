import numpy as np

# inicializo con k-means
def initialize_with_kmeans(X, K, N):
    _, centroids, _ = kmeans(X, K, N)
    weights = np.full(K, 1 / K)
    covariances = np.array([np.cov(X.T) for _ in range(K)])
    return weights, centroids, covariances

def gaussian_density(X, mean, cov):
    d = X.shape[1]
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)
    norm_factor = 1.0 / (np.sqrt((2 * np.pi) ** d * cov_det))
    
    diff = X - mean
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    return norm_factor * np.exp(exponent)

# GMM con EM
def gmm(X, K, max_iters=100, tol=1e-4):
    N, D = X.shape

    weights, means, covariances = initialize_with_kmeans(X, K, N)
    log_likelihood_old = 0

    for iteration in range(max_iters):
        # E-step
        responsibilities = np.zeros((N, K))
        for j in range(K):
            responsibilities[:, j] = weights[j] * gaussian_density(X, means[j], covariances[j])
        
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)

        # M-step
        N_k = responsibilities.sum(axis=0)
        weights = N_k / N
        means = np.dot(responsibilities.T, X) / N_k[:, np.newaxis]
        covariances = np.zeros((K, D, D))
        for j in range(K):
            diff = X - means[j]
            covariances[j] = np.dot((responsibilities[:, j][:, np.newaxis] * diff).T, diff) / N_k[j]

        log_likelihood = np.sum(np.log(np.sum([
            weights[j] * gaussian_density(X, means[j], covariances[j]) for j in range(K)], axis=0)))
        nll = -log_likelihood
        
        if np.abs(nll - log_likelihood_old) < tol:
            break
        log_likelihood_old = nll

    return responsibilities, means, covariances, weights, nll

def initialize_centroids(X, K, N):
    return X[np.random.choice(N, K, replace=False)]

# matriz OHE que clasifica las muestas al cluster
def calculate_r(X, centroids, N):
    K = centroids.shape[0]
    r = np.zeros((N, K))
    for i, point in enumerate(X):
        distances = [np.sum((point - centroid) ** 2) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        r[i, closest_centroid] = 1
    return r

def update_centroids(X, r, K, N):
    new_centroids = []
    for j in range(K):
        points_in_cluster = X[r[:, j] == 1]
        if len(points_in_cluster) > 0:
            new_centroids.append(np.mean(points_in_cluster, axis=0))
        else:
            new_centroids.append(X[np.random.choice(N)])
    return np.array(new_centroids)

def kmeans(X, K, max_iters=100, tol=1e-4):
    N, D = X.shape 
    centroids = initialize_centroids(X, K, N)
    for _ in range(max_iters):
        r = calculate_r(X, centroids, N)
        new_centroids = update_centroids(X, r, K, N)
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    return r, centroids, 0 