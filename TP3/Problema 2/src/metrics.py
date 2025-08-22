import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def precision(y_true, y_pred, clase):
    tp = np.sum((y_true == clase) & (y_pred == clase))  
    fp = np.sum((y_true != clase) & (y_pred == clase)) 
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred, clase):
    tp = np.sum((y_true == clase) & (y_pred == clase)) 
    fn = np.sum((y_true == clase) & (y_pred != clase))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred, clase):
    p = precision(y_true, y_pred, clase)
    r = recall(y_true, y_pred, clase)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

def calcular_metricas(y_true, y_pred):
    clases = np.unique(y_true)
    metrics = {}
    metrics['accuracy'] = accuracy(y_true, y_pred)
    for clase in clases:
        metrics[clase] = {
            'precision': precision(y_true, y_pred, clase),
            'recall': recall(y_true, y_pred, clase),
            'f1_score': f1_score(y_true, y_pred, clase)
        }
    return metrics

def imprimir_metricas(y_true, y_pred):
    metrics = calcular_metricas(y_true, y_pred)
    print(f"{'Clase':<10}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1 Score':<10}")
    print('-' * 50)
    for clase in np.unique(y_true):
        if clase in metrics:
            print(f"{clase:<10}{metrics['accuracy']:<10.4f}{metrics[clase]['precision']:<10.4f}{metrics[clase]['recall']:<10.4f}{metrics[clase]['f1_score']:<10.4f}")

def confusion_matrix(y_true, y_pred, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: i for i, label in enumerate(labels)}
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[label_to_index[true_label], label_to_index[pred_label]] += 1
    return matrix

def plot_confusion_matrix(matrix, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='BuPu', cbar=True,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicciones')
    plt.ylabel('Etiquetas Verdaderas')
    plt.title('Matriz de Confusión')
    plt.show()

def calcular_roc_multiclase(y_true, y_proba, num_classes):
    roc_points = {}
    for clase in range(num_classes):
        y_true_bin = np.where(y_true == clase, 1, 0)
        y_proba_clase = y_proba[:, clase]
        thresholds = np.sort(np.unique(y_proba_clase))[::-1]
        tpr = [] 
        fpr = []  
        for threshold in thresholds:
            y_pred = (y_proba_clase >= threshold).astype(int)
            TP = np.sum((y_pred == 1) & (y_true_bin == 1))
            FP = np.sum((y_pred == 1) & (y_true_bin == 0))
            FN = np.sum((y_pred == 0) & (y_true_bin == 1))
            TN = np.sum((y_pred == 0) & (y_true_bin == 0))
            tpr.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
            fpr.append(FP / (FP + TN) if (FP + TN) > 0 else 0)
        roc_points[clase] = (fpr, tpr)
    return roc_points

def plot_roc_multiclase(roc_points, num_classes):
    plt.figure(figsize=(8, 6))
    for clase in range(num_classes):
        fpr, tpr = roc_points[clase]
        plt.plot(fpr, tpr, label=f'Clase {clase} (vs todos)')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC para múltiples clases')
    plt.legend()
    plt.show()


def calcular_pr_multiclase(y_true, y_proba, num_classes):
    pr_points = {}
    for clase in range(num_classes):
        y_true_bin = np.where(y_true == clase, 1, 0)
        y_proba_clase = y_proba[:, clase]
        thresholds = np.sort(np.unique(y_proba_clase))[::-1]
        precision = []  
        recall = []  
        for threshold in thresholds:
            y_pred = (y_proba_clase >= threshold).astype(int)
            TP = np.sum((y_pred == 1) & (y_true_bin == 1))
            FP = np.sum((y_pred == 1) & (y_true_bin == 0))
            FN = np.sum((y_pred == 0) & (y_true_bin == 1))
            precision.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
            recall.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        pr_points[clase] = (precision, recall)
    return pr_points

def plot_pr_multiclase(pr_points, num_classes):
    plt.figure(figsize=(8, 6))
    for clase in range(num_classes):
        precision, recall = pr_points[clase]
        plt.plot(recall, precision, label=f'Clase {clase} (vs todos)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall para múltiples clases')
    plt.legend()
    plt.show()

def calcular_auc(x, y):
    auc = np.trapz(y, x)
    return auc