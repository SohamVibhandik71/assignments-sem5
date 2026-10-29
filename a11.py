from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models and parameters
models = {
    "SVM": (
        make_pipeline(StandardScaler(), SVC()),
        {
            "svc__C": [0.1, 1, 10],
            "svc__kernel": ["linear", "rbf"]
        }
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10]
        }
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.5, 1.0]
        }
    )
}

# Compare models
for name, (model, params) in models.items():
    print("\n", name)

    # Before optimization: cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    before = accuracy_score(y_test, model.predict(X_test))

    # After optimization: Grid Search with cross-validation
    grid = GridSearchCV(model, params, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    after = accuracy_score(y_test, grid.best_estimator_.predict(X_test))

    print("CV Accuracy before:", round(cv_scores.mean() * 100, 2), "%")
    print("Test Accuracy before:", round(before * 100, 2), "%")
    print("Best Parameters:", grid.best_params_)
    print("Best CV Accuracy after:", round(grid.best_score_ * 100, 2), "%")
    print("Test Accuracy after:", round(after * 100, 2), "%")