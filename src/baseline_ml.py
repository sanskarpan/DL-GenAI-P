"""
Classical ML baseline (Milestone 2).

Pipeline:
1. Create synthetic mashups from training stems (match test distribution)
2. Extract MFCC + spectral features
3. Train LightGBM/XGBoost on balanced classes
4. Evaluate macro F1 on validation set
5. Predict on test set and generate submission

This gives a strong baseline FASTER than CNN training.
Expected performance: 40-60% macro F1 with good augmentation.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# Append parent for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config
from src.config import set_seed
from src.augment import build_stem_index, split_songs_train_val, create_synthetic_mashup, load_audio
from src.features import extract_full_features


def extract_features_from_synthetic(stem_index: dict,
                                     n_samples: int,
                                     add_noise: bool = True,
                                     desc: str = "Extracting features") -> tuple:
    """
    Generate synthetic mashups and extract ML features.

    Returns:
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) integer labels
    """
    genres = list(stem_index.keys())
    n_per_genre = n_samples // len(genres)
    remainder = n_samples % len(genres)

    X_list, y_list = [], []

    for i, genre in enumerate(genres):
        n = n_per_genre + (1 if i < remainder else 0)
        label = config.GENRE_TO_IDX[genre]

        for _ in tqdm(range(n), desc=f"{desc} [{genre}]", leave=False):
            try:
                audio, _ = create_synthetic_mashup(
                    genre=genre,
                    stem_index=stem_index,
                    add_noise=add_noise,
                    apply_tempo=True,
                )
                features = extract_full_features(audio)
                X_list.append(features)
                y_list.append(label)
            except Exception as e:
                print(f"Warning: {genre} mashup failed: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def extract_features_from_test(test_csv_path: Path = config.TEST_CSV,
                                data_dir: Path = config.DATA_DIR) -> tuple:
    """
    Extract features from all test mashup files.
    For long files, use the first 30 seconds.

    Returns:
        X: (n_test, n_features) feature matrix
        ids: list of test IDs
    """
    test_df = pd.read_csv(test_csv_path)
    ids = test_df['id'].astype(str).str.zfill(4).tolist()
    filenames = test_df['filename'].tolist()

    X_list = []

    for fname in tqdm(filenames, desc="Extracting test features"):
        path = data_dir / fname
        try:
            # Load up to 30 seconds for feature extraction
            audio = load_audio(path, sr=config.SAMPLE_RATE, duration=30.0)
            features = extract_full_features(audio)
        except Exception as e:
            print(f"Warning: failed to process {fname}: {e}")
            # Use zeros as fallback
            features = np.zeros(300, dtype=np.float32)  # approximate feature size
        X_list.append(features)

    X = np.array(X_list, dtype=np.float32)
    return X, ids


def train_classical_ml(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        model_type: str = "lgbm") -> object:
    """
    Train classical ML classifier.

    Args:
        model_type: "lgbm", "xgb", "rf", "logreg", or "gbm"
    """
    print(f"\nTraining {model_type.upper()} classifier...")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    if model_type == "lgbm":
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                class_weight='balanced',
                random_state=config.SEED,
                n_jobs=-1,
                verbose=-1,
            )
        except ImportError:
            print("LightGBM not available, falling back to RandomForest")
            model_type = "rf"

    if model_type == "xgb":
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.SEED,
                n_jobs=-1,
                verbosity=0,
            )
        except ImportError:
            print("XGBoost not available, falling back to RandomForest")
            model_type = "rf"

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=config.SEED,
            n_jobs=-1,
        )
    elif model_type == "logreg":
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                random_state=config.SEED,
                n_jobs=-1,
            ))
        ])

    # Fit
    t0 = time.time()
    if model_type in ("lgbm",):
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(100)])
    else:
        model.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"Training took {elapsed:.1f}s")

    # Evaluate
    val_preds = model.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average='macro')
    print(f"\nValidation Macro F1: {val_f1:.4f}")
    print(classification_report(y_val, val_preds,
                                 target_names=config.GENRES, digits=4))

    return model


def run_baseline_pipeline(
        n_train: int = 2000,
        n_val: int = 500,
        model_type: str = "lgbm",
        save_submission: bool = True,
        submission_suffix: str = "_ml_baseline",
) -> dict:
    """
    Run the full classical ML baseline pipeline.

    Args:
        n_train: Number of synthetic training mashups
        n_val: Number of synthetic validation mashups
        model_type: ML model type
        save_submission: Whether to save submission CSV

    Returns:
        dict with results
    """
    set_seed(config.SEED)

    print("=" * 60)
    print("CLASSICAL ML BASELINE PIPELINE")
    print("=" * 60)

    # 1. Build stem index and split
    print("\n1. Building stem index...")
    stem_index = build_stem_index()
    for genre, songs in stem_index.items():
        print(f"   {genre}: {len(songs)} songs")

    train_index, val_index = split_songs_train_val(stem_index)
    print(f"\nSplit: {sum(len(v) for v in train_index.values())} train songs, "
          f"{sum(len(v) for v in val_index.values())} val songs")

    # 2. Generate synthetic training features
    print(f"\n2. Generating {n_train} synthetic training mashups...")
    X_train, y_train = extract_features_from_synthetic(
        train_index, n_train, add_noise=True, desc="Train"
    )
    print(f"   Feature shape: {X_train.shape}")
    print(f"   Label distribution: {np.bincount(y_train)}")

    # 3. Generate synthetic validation features
    print(f"\n3. Generating {n_val} synthetic validation mashups...")
    X_val, y_val = extract_features_from_synthetic(
        val_index, n_val, add_noise=True, desc="Val"
    )

    # 4. Save features for reuse
    features_path = config.FEATURES_DIR / "ml_train_features.npz"
    np.savez(features_path, X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val)
    print(f"\n4. Features saved to {features_path}")

    # 5. Train model
    model = train_classical_ml(X_train, y_train, X_val, y_val, model_type)

    # 6. Save model
    model_path = config.MODELS_DIR / f"baseline_{model_type}.pkl"
    joblib.dump(model, model_path)
    print(f"\n5. Model saved to {model_path}")

    # 7. Extract test features
    print("\n6. Extracting test features...")
    X_test, test_ids = extract_features_from_test()
    print(f"   Test feature shape: {X_test.shape}")

    # 8. Predict
    print("\n7. Generating predictions...")
    test_preds = model.predict(X_test)
    pred_genres = [config.IDX_TO_GENRE[i] for i in test_preds]

    submission_df = pd.DataFrame({"id": test_ids, "genre": pred_genres})

    if save_submission:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = config.SUBMISSIONS_DIR / f"submission_{ts}{submission_suffix}.csv"
        submission_df.to_csv(out_path, index=False)
        print(f"\n8. Submission saved to {out_path}")
        print(f"   Genre distribution:\n{submission_df['genre'].value_counts().to_string()}")

    return {
        "model": model,
        "submission": submission_df,
        "X_train_shape": X_train.shape,
        "X_val_shape": X_val.shape,
    }


if __name__ == "__main__":
    results = run_baseline_pipeline(
        n_train=2000,
        n_val=500,
        model_type="lgbm",
        save_submission=True,
    )
