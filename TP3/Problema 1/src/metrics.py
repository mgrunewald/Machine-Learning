import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc

def precision_score(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    if predicted_positives == 0:
        return 0
    return true_positives / predicted_positives

def recall_score(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    if actual_positives == 0:
        return 0
    return true_positives / actual_positives

def accuracy_score(y_true, y_pred):
    total_predictions = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    if total_predictions == 0:
        return 0
    return correct_predictions / total_predictions

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

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

def plot_precision_recall_curve_s(y_true, y_scores, just_result=bool, ax=None): #Hecho con funciones de sklearn, dado a que el mío funcionaba mal y me estaba retrasando con el trabajo (consultado con Trini)
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auc_pr = auc(recall, precision)
    if just_result:
        return auc_pr
    if ax is None:
        ax = plt.gca() 
    ax.plot(recall, precision, marker='.', label=f'AUC PR = {auc_pr:.2f}')
    ax.set_xlabel('Recall')  
    ax.set_ylabel('Precision') 
    ax.set_title('Curva Precisión-Recall')
    ax.fill_between(recall, precision, alpha=0.2, color='b')
    ax.legend()
    return auc_pr

def roc_curve(y_true, y_scores):
    thresholds = np.sort(y_scores)
    tpr, fpr = [], []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tpr.append(recall_score(y_true, y_pred))  
        fpr.append(np.sum((y_true == 0) & (y_pred == 1)) / (np.sum(y_true == 0) or 1))
    fpr, tpr = zip(*sorted(zip(fpr, tpr)))
    return fpr, tpr, thresholds

def plot_roc_curve(y_true, y_scores, just_result=bool, ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_roc = np.trapz(tpr, fpr)  
    if just_result == True:
        return auc_roc
    if ax is None:
        ax = plt.gca() 
    ax.plot(fpr, tpr, marker='.')
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)') 
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)') 
    ax.set_title('Curva ROC')
    ax.fill_between(fpr, tpr, alpha=0.2, color='b')
    ax.text(0.5, 0.5, f'AUC = {auc_roc:.4f}', fontsize=12, ha='center')
    return auc_roc

def plot_both_curves(y_true, y_scores):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6)) 
    plot_precision_recall_curve_s(y_true, y_scores, just_result=False, ax=ax[0])
    plot_roc_curve(y_true, y_scores, just_result=False, ax=ax[1])
    plt.tight_layout()
    plt.show()