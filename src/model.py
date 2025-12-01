import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from typing import Dict

# --- Baseline Model Training and Evaluation ---

def train_and_evaluate_baseline(X_train: np.ndarray, y_train: np.ndarray, 
                                X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Trains the Logistic Regression baseline model on training data and
    evaluates its performance on the separate test data.
    
    This function handles the entire process for your baseline classifier.
    """
    
    # 1. Train the Model (Fit ONLY on Training Data)
    # Corrected the typo: 'liblinear' is a good choice for smaller datasets.
    model = LogisticRegression(
        solver='liblinear', 
        C=1.0, 
        random_state=42,
        max_iter=1000
    )
    model.fit(X_train, y_train)
    
    # 2. Predict Probabilities for Evaluation (Use ONLY Test Data)
    # We use predict_proba for the crucial AUC metric.
    y_pred_proba = model.predict_proba(X_test)[:, 1] 
    y_pred_class = model.predict(X_test)
    
    # 3. Calculate Performance Metrics (Crucial for BCI)
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred_class)
    
    # Confusion Matrix for detailed performance breakdown
    # tn, fp, fn, tp will be crucial if the dataset is imbalanced.
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
    
    # 4. Return Results
    results = {
        'AUC': auc,
        'Accuracy': accuracy,
        'True Positives (TP)': int(tp),
        'False Positives (FP)': int(fp),
        'True Negatives (TN)': int(tn),
        'False Negatives (FN)': int(fn),
    }
    
    return results

# --- Transformer Model (Placeholder for Phase 4) ---

# class BCITransformer(torch.nn.Module):
#     # ... (Define your deep learning model architecture here)
#     pass