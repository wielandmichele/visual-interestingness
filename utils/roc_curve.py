from sklearn.metrics import roc_auc_score, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List

def get_roc(title: str, path: str, y_prob: List[float], y_test: List[int], label: str) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = round(roc_auc_score(y_test,y_prob),3)
    label = label+" "+str(auc)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=label)
    plt.xlabel('1-Specifity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig(path, dpi = 200)

def get_roc_multiclass(test_labels: List[int], y_pred: List[float], target_names: List[str], img_path: str) -> None:
    fpr, tpr, roc_auc = dict(), dict(), dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fig, ax = plt.subplots(figsize=(10, 8))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    num_classes = len(target_names)
    color_map = cm.get_cmap('tab10')
    for class_id in range(num_classes):
        color = color_map(class_id / num_classes)
        RocCurveDisplay.from_predictions(
            test_labels[:, class_id],
            y_pred[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
    )

    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve One-vs-Rest")
    plt.legend()
    plt.savefig(img_path, dpi = 200)
    plt.show()
        
