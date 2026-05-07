"""
AI_stats_lab.py

Lab: Training and Evaluating Classification Models

Topics:
- Confusion matrix
- Recall
- Fallout
- Precision
- Accuracy
- Thresholding prediction scores
- Effect of changing threshold
- Training two classifiers
- Comparing model performance

Instructions:
- Implement all functions.
- Do NOT change function names.
- Do NOT print inside functions.
- Return exactly the required formats.
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Question 1: Confusion Matrix, Metrics, and Threshold Effects
# ============================================================

def confusion_matrix_counts(y_true, y_pred):
    """
    Compute confusion matrix counts for binary classification.

    Parameters:
        y_true : true labels, values must be 0 or 1
        y_pred : predicted labels, values must be 0 or 1

    Returns:
        (TP, FP, FN, TN)

    Definitions:
        TP = actual positive and predicted positive
        FP = actual negative but predicted positive
        FN = actual positive but predicted negative
        TN = actual negative and predicted negative

    Hints:
        - Compare y_true and y_pred element by element.
        - You may use a loop or NumPy boolean masks.
        - Return the values in this exact order:
              (TP, FP, FN, TN)
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    return (TP, FP, FN, TN) 


def classification_metrics(y_true, y_pred):
    """
    Compute classification metrics.

    Returns:
        A dictionary with keys:
            "recall"
            "fallout"
            "precision"
            "accuracy"

    Formulas:
        recall    = TP / (TP + FN)
        fallout   = FP / (FP + TN)
        precision = TP / (TP + FP)
        accuracy  = (TP + TN) / (TP + FP + FN + TN)

    If a denominator is zero, return 0.0 for that metric.

    Hints:
        - First call confusion_matrix_counts.
        - Then compute each metric from TP, FP, FN, and TN.
        - Return a dictionary, not a tuple or list.
    """
    TP, FP, FN, TN = confusion_matrix_counts(y_true, y_pred)

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fallout = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0

    return {
        "recall": recall,
        "fallout": fallout,
        "precision": precision,
        "accuracy": accuracy,
    }


def apply_threshold(scores, threshold):
    """
    Convert prediction scores into binary predictions.

    Parameters:
        scores    : predicted probabilities or scores
        threshold : decision threshold

    Returns:
        NumPy array:
            1 if score >= threshold
            0 otherwise

    Hints:
        - Convert scores to a NumPy array if needed.
        - Use the rule:
              score >= threshold  -> 1
              score < threshold   -> 0
    """
    scores = np.array(scores)
    pred = (scores >= threshold).astype(int)
    return pred


def threshold_metrics_analysis(y_true, scores, thresholds):
    """
    Analyze how changing threshold affects recall and fallout.

    Parameters:
        y_true     : true binary labels
        scores     : predicted probabilities or scores
        thresholds : list or array of threshold values

    Returns:
        A list of dictionaries, one per threshold:

        [
            {
                "threshold": threshold,
                "recall": recall,
                "fallout": fallout,
                "precision": precision,
                "accuracy": accuracy
            },
            ...
        ]

    Hints:
        - For each threshold:
            1. Convert scores to predictions using apply_threshold.
            2. Compute classification_metrics.
            3. Store the threshold and metrics in a dictionary.

    Important idea:
        Lower threshold usually predicts more positives.
        This usually increases recall but may also increase fallout.

        Higher threshold usually predicts fewer positives.
        This usually decreases fallout but may also decrease recall.
    """
    results = []
    for threshold in thresholds:
        y_pred = apply_threshold(scores, threshold)
        metrics = classification_metrics(y_true, y_pred)

        results.append(
            {
                "threshold": threshold,
                "recall": metrics["recall"],
                "fallout": metrics["fallout"],
                "precision": metrics["precision"],
                "accuracy": metrics["accuracy"],
            }
        )
    return results


# ============================================================
# Question 2: Train Two Classifiers and Evaluate Them
# ============================================================

def train_two_classifiers(X_train, y_train):
    """
    Train two binary classifiers:

    1. Logistic Regression
    2. Decision Tree Classifier

    Parameters:
        X_train : training features
        y_train : training labels

    Returns:
        A dictionary:

        {
            "logistic_regression": trained logistic regression model,
            "decision_tree": trained decision tree model
        }

    Requirements:
        - Use LogisticRegression from sklearn.linear_model
        - Use DecisionTreeClassifier from sklearn.tree
        - Use max_iter=1000 for LogisticRegression
        - Use random_state=0 for DecisionTreeClassifier

    Hints:
        - Create both models.
        - Fit both models using model.fit(X_train, y_train).
        - Return the trained models in a dictionary.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)

    return {
        "logistic_regression": lr,
        "decision_tree": dt
    }


def evaluate_classifier(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained classifier.

    Steps:
        1. Get predicted probabilities for the positive class.
        2. Convert probabilities to predicted labels using threshold.
        3. Compute confusion matrix counts.
        4. Compute recall, fallout, precision, and accuracy.

    Returns:
        A dictionary with keys:

        {
            "TP": value,
            "FP": value,
            "FN": value,
            "TN": value,
            "recall": value,
            "fallout": value,
            "precision": value,
            "accuracy": value
        }

    Hints:
        - Use:
              model.predict_proba(X_test)[:, 1]
          to get positive-class probabilities.
        - Then call apply_threshold.
        - Then call confusion_matrix_counts.
        - Then call classification_metrics.
        - Combine the counts and metrics into one dictionary.
    """
    prob = model.predict_proba(X_test)[:, 1]
    pred = apply_threshold(prob, threshold)
    TP, FP, FN, TN = confusion_matrix_counts(y_test, pred)
    metrics = classification_metrics(y_test, pred)
    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "recall": metrics["recall"],
        "fallout": metrics["fallout"],
        "precision": metrics["precision"],
        "accuracy": metrics["accuracy"],
    }


def compare_classifiers(X_train, y_train, X_test, y_test, threshold=0.5):
    """
    Train two classifiers and evaluate both on the same test set.

    Returns:
        A dictionary:

        {
            "logistic_regression": evaluation_dictionary,
            "decision_tree": evaluation_dictionary
        }

    Each evaluation dictionary must contain:
        TP, FP, FN, TN, recall, fallout, precision, accuracy

    Hints:
        - First call train_two_classifiers.
        - Then evaluate both classifiers using evaluate_classifier.
        - Return a dictionary with results for both models.
    """
    models = train_two_classifiers(X_train, y_train)
    lr_eval = evaluate_classifier(models["logistic_regression"], X_test, y_test, threshold)
    dt_eval = evaluate_classifier(models["decision_tree"], X_test, y_test, threshold)
    return {
        "logistic_regression": lr_eval,
        "decision_tree": dt_eval,
    }


if __name__ == "__main__":
    print("Implement all required functions.")
