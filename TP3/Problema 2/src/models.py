import numpy as np
from collections import Counter

class LinearDiscriminantAnalysis:
    def __init__(self):
        self.m = None  
        self.proyeccion = None 
        self.classes = None 
    
    def calcular_medias_por_clase(self, X, y):
        clases = np.unique(y)
        medias_por_clase = {}
        for clase in clases:
            medias_por_clase[clase] = np.mean(X[y == clase], axis=0)
        return medias_por_clase

    def calcular_covarianza(self, X, y, medias_por_clase):
        n_features = X.shape[1]
        covarianza_intra = np.zeros((n_features, n_features))
        for clase, media in medias_por_clase.items():
            X_clase = X[y == clase] - media
            covarianza_intra += np.dot(X_clase.T, X_clase)
        return covarianza_intra / (len(X) - len(medias_por_clase))
    
    def proyeccion_lda(self, X, y, medias_por_clase, covarianza_intra):
        clases = np.unique(y)
        n_features = X.shape[1]
        covarianza_entre = np.zeros((n_features, n_features))
        mean_total = np.mean(X, axis=0)
        for clase, media in medias_por_clase.items():
            n_clase = X[y == clase].shape[0]
            mean_diff = np.array(media - mean_total).reshape(n_features, 1)
            covarianza_entre += n_clase * np.dot(mean_diff, mean_diff.T)
        covarianza_total = np.linalg.inv(covarianza_intra).dot(covarianza_entre)
        eigenvalues, eigenvectors = np.linalg.eigh(covarianza_total)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        return eigenvectors[:, :len(clases)-1]

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.m = self.calcular_medias_por_clase(X, y)
        covarianza_intra = self.calcular_covarianza(X, y, self.m)
        self.proyeccion = self.proyeccion_lda(X, y, self.m, covarianza_intra)
    
    def predict(self, X):
        predicciones = []
        for x in X:
            proyeccion_x = x.dot(self.proyeccion)
            distancias = {}
            for clase, media in self.m.items():
                proyeccion_media = media.dot(self.proyeccion)
                distancias[clase] = np.linalg.norm(proyeccion_x - proyeccion_media)
            predicciones.append(min(distancias, key=distancias.get))
        return np.array(predicciones)
    
    def predict_proba(self, X):
        """
        Calcula las probabilidades basadas en la función de densidad de probabilidad Gaussiana.
        """
        probas = []
        for x in X:
            proyeccion_x = x.dot(self.proyeccion)
            probabilidades_clase = []
            for clase, media in self.m.items():
                proyeccion_media = media.dot(self.proyeccion)
                dist = np.linalg.norm(proyeccion_x - proyeccion_media)
                prob_clase = np.exp(-0.5 * dist**2)  # Simplificación Gaussiana
                probabilidades_clase.append(prob_clase)
            probabilidades_clase = np.array(probabilidades_clase)
            probabilidades_clase /= np.sum(probabilidades_clase)
            probas.append(probabilidades_clase)
        return np.array(probas)


class LogisticRegressionMulticlass:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_reg=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
    
    def one_hot_encode(self, y):
        y = y.astype(int)
        classes = np.unique(y)
        one_hot = np.zeros((y.size, classes.size))
        one_hot[np.arange(y.size), y] = 1
        return one_hot
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        self.weights = np.random.randn(num_features, num_classes) * 0.01
        self.bias = np.zeros((1, num_classes))
        y_one_hot = self.one_hot_encode(y)
        for _ in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(z)
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y_one_hot)) + (self.lambda_reg / num_samples) * self.weights
            db = (1 / num_samples) * np.sum(y_pred - y_one_hot, axis=0, keepdims=True)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(z)
        return np.argmax(y_pred, axis=1) 
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)


class Node():
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.feature_importance = self.data.shape[0] * self.information_gain
        self.left = None
        self.right = None

class DecisionTree():
    def __init__(self, 
                 max_depth=4, 
                 min_samples_leaf=1, 
                 min_information_gain=0.0) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain

    def entropy(self, class_probabilities: list) -> float:
        return sum([-p * np.log2(p) for p in class_probabilities if p>0])
    
    def class_probabilities(self, labels: list) -> list:
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def data_entropy(self, labels: list) -> float:
        return self.entropy(self.class_probabilities(labels))
    
    def partition_entropy(self, subsets: list) -> float:
        total_count = sum([len(subset) for subset in subsets]) 
        return sum([self.data_entropy(subset) * (len(subset) / total_count) for subset in subsets])
    
    def split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]
        return group1, group2
        
    def find_best_split(self, data: np.array) -> tuple:
        min_part_entropy = 1e9
        feature_idx =  list(range(data.shape[1]-1))
        for idx in feature_idx: 
            feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25)) 
            for feature_val in feature_vals: 
                g1, g2, = self.split(data, idx, feature_val)
                part_entropy = self.partition_entropy([g1[:, -1], g2[:, -1]]) 
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx
                    min_entropy_feature_val = feature_val
                    g1_min, g2_min = g1, g2
        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def find_label_probs(self, data: np.array) -> np.array:
        labels_as_integers = data[:,-1].astype(int)
        total_labels = len(labels_as_integers)
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def create_tree(self, data: np.array, current_depth: int) -> Node:
        if current_depth > self.max_depth:
            return None
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self.find_best_split(data)
        label_probabilities = self.find_label_probs(data)
        node_entropy = self.entropy(label_probabilities)
        information_gain = node_entropy - split_entropy
        node = Node(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        elif information_gain < self.min_information_gain:
            return node
        current_depth += 1
        node.left = self.create_tree(split_1_data, current_depth)
        node.right = self.create_tree(split_2_data, current_depth)
    
        return node
    
    def predict_one_sample(self, X: np.array) -> np.array:
        node = self.tree
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)
        self.tree = self.create_tree(data=train_data, current_depth=0)

    def predict_proba(self, X_set: np.array) -> np.array:
        pred_probs = np.apply_along_axis(self.predict_one_sample, 1, X_set)
        
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        return preds   
    
class RandomForest:
    def __init__(self, n_trees=10, max_depth=4, min_samples_leaf=1, min_information_gain=0.0, sample_size_ratio=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.sample_size_ratio = sample_size_ratio
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = int(X.shape[0] * self.sample_size_ratio)
        indices = np.random.choice(X.shape[0], size=n_samples, replace=True)
        return X[indices], y[indices]

    def train(self, X_train, y_train):
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap_sample(X_train, y_train)
            tree = DecisionTree(max_depth=self.max_depth, 
                                min_samples_leaf=self.min_samples_leaf, 
                                min_information_gain=self.min_information_gain)
            tree.train(X_sample, y_sample)
            self.trees.append(tree)

    def predict_proba(self, X_set):
        tree_pred_probs = []
        for tree in self.trees:
            tree_pred_probs.append(tree.predict_proba(X_set))
        tree_pred_probs = np.array(tree_pred_probs)
        averaged_probs = np.mean(tree_pred_probs, axis=0)
        return averaged_probs

    def predict(self, X_set):
        averaged_probs = self.predict_proba(X_set)
        final_predictions = np.argmax(averaged_probs, axis=1)
        return final_predictions
