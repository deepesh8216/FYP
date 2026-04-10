from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=["Fake", "Real"]),
    }


def pretty_print_metrics(metrics):
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"F1 Macro       : {metrics['f1_macro']:.4f}")
    print(f"Precision Macro: {metrics['precision_macro']:.4f}")
    print(f"Recall Macro   : {metrics['recall_macro']:.4f}")
    print("\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"Actual Fake: [{cm[0][0]:4d}  {cm[0][1]:4d}]")
    print(f"Actual Real: [{cm[1][0]:4d}  {cm[1][1]:4d}]")
    print("\nDetailed Report:")
    print(metrics["classification_report"])
