"""
classifier.py
-------------
Train, persist, and load a scikit-learn RandomForestClassifier that maps
per-cell terrain features -> terrain class (0=Flat/Sand, 1=Rocky,
2=Steep/Terrain Hazard, 3=Obstacle).

Pipeline implemented here:

    feature array (N,4) + label array (N,)
                       |
                       v
             split_dataset()  -> train / val / test splits (stratified)
                       |
                       v
             train_classifier() -> fitted RandomForestClassifier
                       |
                       v
             save_model() / load_model()  -> models/terrain_classifier.pkl
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FEATURE_ORDER = ("depth", "slope", "roughness", "curvature")


@dataclass
class DatasetSplits:
    """Train / validation / test splits of features and labels."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def __str__(self) -> str:
        return (
            "Dataset Splits\n"
            "---------------\n"
            f"train: {self.X_train.shape[0]} cells\n"
            f"val:   {self.X_val.shape[0]} cells\n"
            f"test:  {self.X_test.shape[0]} cells"
        )


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_split: float,
    val_split: float,
    test_split: float,
    random_seed: int = 42,
) -> DatasetSplits:
    """
    Split feature/label arrays into stratified train/val/test sets.

    Input:
        X: (N, F) feature array.
        y: (N,) integer class labels.
        train_split, val_split, test_split: fractions from config; must sum to ~1.0.
        random_seed: seed for reproducible splitting (config: project.random_seed).
    Processing:
        Two-stage stratified split: first carve off `test_split` for the
        test set, then split the remainder into train/val in the correct
        relative proportion. Stratification keeps class balance similar
        across all three splits (important since Obstacle is a minority
        class at ~4.6% of cells).
    Output:
        DatasetSplits with X_train/y_train, X_val/y_val, X_test/y_test.
    """
    total = train_split + val_split + test_split
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"train/val/test splits must sum to 1.0, got {total}")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_seed, stratify=y
    )
    # val_split as a fraction of the *remaining* trainval portion
    relative_val = val_split / (train_split + val_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val, random_state=random_seed, stratify=y_trainval
    )

    return DatasetSplits(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_seed: int = 42,
) -> RandomForestClassifier:
    """
    Fit a RandomForestClassifier on the training split.

    Input:
        X_train: (N, F) training features.
        y_train: (N,) training labels.
        n_estimators, max_depth: from config.terrain.classifier.
        random_seed: from config.project.random_seed (also used as the
                     forest's random_state for reproducibility).
    Processing:
        Instantiates and fits sklearn's RandomForestClassifier. class_weight
        is set to "balanced" to compensate for the Obstacle class being a
        minority (~4.6% of labeled cells) so the forest doesn't collapse to
        always predicting the majority Flat/Sand class.
    Output:
        Fitted RandomForestClassifier.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """
    Persist a fitted model to disk with joblib.

    Input:
        model: fitted RandomForestClassifier.
        model_path: destination path (config.terrain.model_path); parent
                    directories are created if missing.
    Processing:
        joblib.dump to `model_path`.
    Output:
        None. Writes a file at `model_path`.
    """
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(model_path: str) -> RandomForestClassifier:
    """
    Load a previously saved model from disk.

    Input:
        model_path: path written by save_model().
    Processing:
        joblib.load, with a clear error if the file does not exist.
    Output:
        Fitted RandomForestClassifier.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Run src/terrain/run_pipeline.py first to train one."
        )
    return joblib.load(path)


def predict_grid(model: RandomForestClassifier, X: np.ndarray, grid_shape: tuple) -> np.ndarray:
    """
    Predict terrain classes for every cell and reshape back to a grid.

    Input:
        model: fitted RandomForestClassifier.
        X: (N, F) feature array (N == grid_shape[0] * grid_shape[1]).
        grid_shape: (rows, cols) of the original bathymetry grid.
    Processing:
        model.predict(X), then reshape (N,) -> grid_shape.
    Output:
        2D int array of predicted class ids, shape == grid_shape.
    """
    preds = model.predict(X)
    return preds.reshape(grid_shape)
