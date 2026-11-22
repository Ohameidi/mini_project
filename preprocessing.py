from pathlib import Path
import numpy as np
import mne
from mne.datasets import eegbci
from mne.decoding import CSP

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def load_task4_subject_raw(
    subject: int = 1,
    runs: list[int] | None = None,
    dataset_path: str = "/home/om/Documents/Stud/AML/mini_project/datasets",
) -> mne.io.BaseRaw:
    """Load and concatenate EEGBCI Task 4 runs (6, 10, 14) for a single subject."""
    if runs is None:
        runs = [6, 10, 14]

    target_dir = Path(dataset_path).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    file_paths = eegbci.load_data([subject], runs, path=str(target_dir), verbose=False)
    raw_list = []

    for fp in file_paths:
        raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
        eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore")
        raw_list.append(raw)

    raw_concat = mne.concatenate_raws(raw_list, verbose=False)
    return raw_concat


def preprocess_and_epoch(
    raw: mne.io.BaseRaw,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    tmin: float = 0.0,
    tmax: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, mne.Epochs]:
    """Filter continuous EEG data and extract motor imagery epochs (T1 vs. T2)."""
    raw_filtered = raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose=False,
    )

    events, event_id_all = mne.events_from_annotations(raw_filtered, verbose=False)

    # In EEGBCI Task 4 (runs 6, 10, 14): T1 = both fists, T2 = both feet
    target_event_id = {
        k: v for k, v in event_id_all.items() if k in ("T1", "T2")
    }
    if set(target_event_id.keys()) != {"T1", "T2"}:
        raise RuntimeError(f"Expected T1 and T2 annotations, got: {event_id_all}")

    epochs = mne.Epochs(
        raw_filtered,
        events=events,
        event_id=target_event_id,
        tmin=tmin,
        tmax=tmax,
        proj=True,
        picks="eeg",
        baseline=None,
        preload=True,
        verbose=False,
    )

    # X shape: (n_epochs, n_channels, n_times); y shape: (n_epochs,)
    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1]
    return X, y, epochs


def train_and_evaluate_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Split data, tune CSP + LogisticRegression hyperparameters, and evaluate."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )

    param_grid = {
        "csp__n_components": [4, 6, 8],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Map binary labels to {0, 1} for ROC-AUC computation
    classes_sorted = np.sort(np.unique(y))
    y_test_bin = (y_test == classes_sorted[1]).astype(int)

    metrics = {
        "best_params": grid_search.best_params_,
        "cv_best_accuracy": float(grid_search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test_bin, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["T1 (Both Fists)", "T2 (Both Feet)"],
            digits=3,
        ),
        "best_model": best_model,
    }
    return metrics


def main() -> None:
    subject_id = 1
    task4_runs = [6, 10, 14]
    dataset_dir = "/home/om/Documents/Stud/AML/mini_project/datasets"

    print(f"Loading EEGBCI Subject {subject_id}, Task 4 runs {task4_runs}...")
    raw = load_task4_subject_raw(
        subject=subject_id,
        runs=task4_runs,
        dataset_path=dataset_dir,
    )

    print("Preprocessing (8-30 Hz FIR filter) and extracting epochs...")
    X, y, epochs = preprocess_and_epoch(
        raw,
        l_freq=8.0,
        h_freq=30.0,
        tmin=0.0,
        tmax=4.0,
    )
    print(f"Epochs shape: {X.shape} (trials, channels, timepoints)")
    print(f"Class distribution (T1 vs T2): {np.bincount(y)}")

    print("Training pipeline with hyperparameter tuning (GridSearchCV)...")
    results = train_and_evaluate_pipeline(X, y, test_size=0.2, random_state=42)

    print("\n=== Results ===")
    print("Best hyperparameters:", results["best_params"])
    print(f"5-Fold CV Accuracy (Train): {results['cv_best_accuracy']:.3f}")
    print(f"Test Accuracy:              {results['test_accuracy']:.3f}")
    print(f"Test Balanced Accuracy:     {results['test_balanced_accuracy']:.3f}")
    print(f"Test ROC-AUC:               {results['test_roc_auc']:.3f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(results["confusion_matrix"])
    print("\nClassification Report:")
    print(results["classification_report"])


if __name__ == "__main__":
    main()