import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import homogeneity_score, silhouette_score


def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=True,
            xticklabels=labels, yticklabels=labels, cmap=plt.cm.Blues)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title + ' - Confusion Matrix')
        
def plot_roc(y_true, y_score, title):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    roc_auc = roc_auc_score(y_true, y_score)
    
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + ' - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    
def clustring_report(X, y_true, y_pred):
    homogeneity = homogeneity_score(y_true, y_pred)
    silhouette = silhouette_score(X, y_pred)
    print(f'\u25CF Homogeneity: {homogeneity:.2f}')
    print(f'\u25CF Silhouette:  {silhouette:.2f}')
