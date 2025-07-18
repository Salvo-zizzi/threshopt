# threshopt

[![PyPI version](https://img.shields.io/pypi/v/threshopt.svg)](https://pypi.org/project/threshopt/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
![GitHub last commit](https://img.shields.io/github/last-commit/Salvo-zizzi/threshopt)

**Threshold Optimization Library for Binary Classification**

`threshopt` is a lightweight Python library designed to help find the optimal decision threshold for binary classifiers, improving model performance by customizing the threshold instead of relying on the default 0.5.

------------------------------------------------------------------------

## Features

-   Optimize decision thresholds based on any metric (e.g. accuracy, F1-score, G-Mean, Youden’s J)
-   Supports cross-validation threshold optimization for robust model tuning
-   Easy integration with any scikit-learn compatible model
-   Built-in common metrics and ability to use custom metrics
-   Visualize confusion matrices and prediction score distributions (optional)

------------------------------------------------------------------------

## Installation

`bash pip install -e .`

Install in editable mode from the project root directory.

## Quickstart

``` python
from threshopt import optimize_threshold, optimize_threshold_cv, gmean_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Optimize threshold on the test set
best_thresh, best_val = optimize_threshold(model, X, y, metric=f1_score)
print(f"Best threshold: {best_thresh:.2f}, F1-score: {best_val:.4f}")

# Optimize threshold with cross-validation
best_thresh_cv, best_val_cv = optimize_threshold_cv(model, X, y, metric=gmean_score, cv=5)
print(f"CV best threshold: {best_thresh_cv:.2f}, CV best metric: {best_val_cv:.4f}")


# Load data

data = load_breast_cancer() X, y = data.data, data.target

# Train model

model = RandomForestClassifier(random_state=42) model.fit(X, y)

# Optimize threshold on the test set

best_thresh, best_val = optimize_threshold(model, X, y, metric=f1_score) print(f"Best threshold: {best_thresh:.2f}, F1-score: {best_val:.4f}")

# Optimize threshold with cross-validation

best_thresh_cv, best_val_cv = optimize_threshold_cv(model, X, y, metric=gmean_score, cv=5) print(f"CV best threshold: {best_thresh_cv:.2f}, CV best metric: {best_val_cv:.4f}")
```

## Metrics

Included metrics:

-   `gmean_score`: Geometric Mean of sensitivity and specificity
-   `youden_j_stat`: Youden’s J statistic (sensitivity + specificity - 1)
-   `balanced_acc_score`: Balanced Accuracy (wrapper around scikit-learn)

You can also pass any metric function with signature `metric(y_true, y_pred)`.

------------------------------------------------------------------------

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

------------------------------------------------------------------------

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
