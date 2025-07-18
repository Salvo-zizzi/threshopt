from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

def optimize_threshold(model, X, y_true, metric, plot=True, cm=True, report=True):
    """
    Trova la soglia ottimale per un classificatore binario ottimizzando la metrica specificata.
    
    Args:
        model: modello già addestrato con metodo predict_proba o decision_function
        X: array-like features di test
        y_true: array-like true labels binari
        metric: funzione metrica con firma metric(y_true, y_pred) -> float
        plot: se True, disegna la distribuzione delle predizioni probabilistiche
        cm: se True, stampa la matrice di confusione con soglia ottimale
        report: se True, stampa classification report
    
    Returns:
        best_threshold: soglia ottimale trovata
        best_metric_value: valore della metrica alla soglia ottimale
    """
    
    # Ottieni score probabilistici o decision function normalizzati
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X)
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    else:
        raise ValueError("Model does not support predict_proba or decision_function")
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    thresholds = np.append(thresholds, 1.0)
    
    metric_values = []
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        score = metric(y_true, y_pred)
        metric_values.append(score)
    
    best_idx = np.argmax(metric_values)
    best_threshold = thresholds[best_idx]
    best_metric_value = metric_values[best_idx]
    
    print(f"Best {metric.__name__}: {best_metric_value:.4f} at threshold: {best_threshold:.2f}")
    
    y_pred_best = (y_scores >= best_threshold).astype(int)
    
    if report:
        print(classification_report(y_true, y_pred_best))
    
    if cm:
        cmatrix = confusion_matrix(y_true, y_pred_best)
        disp = ConfusionMatrixDisplay(confusion_matrix=cmatrix)
        disp.plot(cmap=plt.cm.Greens)
        plt.title(f"Confusion Matrix at threshold={best_threshold:.2f}")
        plt.show()
    
    if plot:
        plt.figure(figsize=(8,4))
        plt.hist(y_scores[y_true == 0], bins=30, alpha=0.5, label='Classe 0')
        plt.hist(y_scores[y_true == 1], bins=30, alpha=0.5, label='Classe 1')
        plt.axvline(best_threshold, color='red', linestyle='--', label='Soglia ottimale')
        plt.xlabel('Score probabilità classe 1')
        plt.ylabel('Frequenza')
        plt.title('Distribuzione score predetti')
        plt.legend()
        plt.show()
    
    return best_threshold, best_metric_value
