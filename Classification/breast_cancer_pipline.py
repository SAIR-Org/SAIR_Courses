# enhanced_breast_cancer_pipeline.py

"""
Enhanced End-to-End Breast Cancer Classification Pipeline using scikit-learn

Includes:
- Loading dataset
- Train/test split
- Standardization (Pipeline)
- Logistic Regression with hyperparameter tuning (GridSearchCV)
- Loss curve visualization
- ROC curve & AUC
- Cross-validation evaluation
- Evaluation (accuracy, classification report, confusion matrix)
- Feature importance visualization
"""

# --- 1️⃣ Imports ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score, log_loss
)
from sklearn.pipeline import Pipeline

# --- 2️⃣ Load dataset ---
def load_data():
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    print("📊 Breast Cancer Dataset Info:")
    print(f"Features ({X.shape[1]}): {bc.feature_names}")
    print(f"Classes: {bc.target_names}")
    print(f"Samples: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")
    return X, y, bc

# --- 3️⃣ Preprocess data ---
def preprocess_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

# --- 4️⃣ Build pipeline + hyperparameter tuning ---
def train_model(X_train, y_train, cv_splits=5):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=1000, solver='lbfgs'))
    ])

    param_grid = {
        'logreg__C': [0.01, 0.1, 1, 10],
        'logreg__penalty': ['l2']
    }

    grid = GridSearchCV(pipeline, param_grid, cv=cv_splits, scoring='accuracy')
    grid.fit(X_train, y_train)

    print("\n✅ Best Hyperparameters:", grid.best_params_)
    print("✅ Best CV Accuracy: {:.2f}%".format(grid.best_score_ * 100))

    # Track loss curve manually using log_loss
    model = grid.best_estimator_
    losses = []
    max_iter = 200
    logreg = LogisticRegression(
        C=grid.best_params_['logreg__C'], max_iter=1, solver='lbfgs', warm_start=True
    )
    X_scaled = model.named_steps['scaler'].transform(X_train)
    y_train_arr = np.array(y_train)
    for i in range(max_iter):
        logreg.fit(X_scaled, y_train_arr)
        y_prob = logreg.predict_proba(X_scaled)
        losses.append(log_loss(y_train_arr, y_prob))

    return model, losses

# --- 5️⃣ Evaluate model ---
def evaluate_model(model, X_test, y_test, bc):
    X_scaled = model.named_steps['scaler'].transform(X_test)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\n🎯 Test Accuracy: {acc:.2f}%")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=bc.target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix - Breast Cancer')
    plt.colorbar()
    tick_marks = np.arange(len(bc.target_names))
    plt.xticks(tick_marks, bc.target_names)
    plt.yticks(tick_marks, bc.target_names)
    
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i,j],'d'),
                     ha='center', va='center',
                     color='white' if cm[i,j] > thresh else 'black',
                     fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # ROC Curve
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_score:.2f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Breast Cancer")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return y_pred

# --- 6️⃣ Feature importance ---
def plot_feature_importance(model, bc, top_n=10):
    coef = model.named_steps['logreg'].coef_[0]
    importance = np.abs(coef)
    sorted_idx = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12,6))
    plt.barh(range(len(importance)), importance[sorted_idx], color='steelblue')
    plt.yticks(range(len(importance)), [bc.feature_names[i] for i in sorted_idx])
    plt.xlabel('Absolute Weight (Importance)')
    plt.title('📊 Feature Importance - Logistic Regression')
    plt.grid(True, alpha=0.3, axis='x')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print("Top 5 most important features:")
    for i in range(top_n if top_n <= len(importance) else len(importance)):
        idx = sorted_idx[i]
        print(f"{i+1}. {bc.feature_names[idx]}: weight = {coef[idx]:.4f}")

# --- 7️⃣ Loss curve ---
def plot_loss_curve(losses):
    plt.figure(figsize=(7,5))
    plt.plot(range(len(losses)), losses, color='red')
    plt.xlabel("Iterations")
    plt.ylabel("Log Loss")
    plt.title("📉 Logistic Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- 8️⃣ Cross-validation ---
def cross_val_evaluation(model, X, y, cv_splits=5):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"\n📊 Cross-Validation Accuracy: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

# --- 9️⃣ Main execution ---
if __name__ == "__main__":
    X, y, bc = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model, losses = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test, bc)
    plot_feature_importance(model, bc, top_n=5)
    plot_loss_curve(losses)
    cross_val_evaluation(model, X, y)
