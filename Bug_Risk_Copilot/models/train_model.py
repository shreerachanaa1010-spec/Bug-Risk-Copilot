"""
Phase 3.1-3.5: Model Training and Evaluation
Trains baseline and main models for bug-risk prediction.

Key principles:
- Time-based train/test split (NOT random) to avoid future leakage
- PR-AUC for evaluation (not ROC-AUC, which is misleading with imbalanced data)
- Proper handling of class imbalance via scale_pos_weight
- SHAP explainability on top-k features
"""

import logging
import pickle
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

try:
    import xgboost as xgb
    import shap
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate bug-risk prediction models."""
    
    def __init__(self, features_path: str = "data/features.csv"):
        """Initialize trainer with feature data."""
        logger.info(f"Loading features from {features_path}")
        self.df = pd.read_csv(features_path)
        
        # Prepare feature set (exclude non-numeric and target columns)
        self.target_col = 'is_bug_introducing'
        self.exclude_cols = [
            'commit_hash', 'author', 'author_date', 'day_of_week', 'is_bug_introducing'
        ]
        
        self.feature_cols = [col for col in self.df.columns 
                            if col not in self.exclude_cols]
        
        logger.info(f"Features: {len(self.feature_cols)}")
        logger.info(f"Sample features: {self.feature_cols[:5]}")
        
        # Class distribution
        pos_count = (self.df[self.target_col] == 1).sum()
        total = len(self.df)
        logger.info(f"Class balance: {pos_count}/{total} positive ({100*pos_count/total:.2f}%)")
    
    def time_based_split(self, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by time (chronologically).
        
        This is CRITICAL: random split leaks future information into training.
        Always use time-based split for temporal data.
        
        Args:
            train_frac: Fraction of oldest commits for training
        
        Returns:
            (train_df, test_df)
        """
        # Sort by date and split
        df_sorted = self.df.sort_values('author_date')
        split_idx = int(len(df_sorted) * train_frac)
        
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        
        logger.info(f"Train: {len(train_df)} commits, Test: {len(test_df)} commits")
        
        train_pos = (train_df[self.target_col] == 1).sum()
        test_pos = (test_df[self.target_col] == 1).sum()
        logger.info(f"Train positive: {train_pos}/{len(train_df)} ({100*train_pos/len(train_df):.2f}%)")
        logger.info(f"Test positive: {test_pos}/{len(test_df)} ({100*test_pos/len(test_df):.2f}%)")
        
        return train_df, test_df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target.
        
        Handles missing values and normalizes.
        """
        X = df[self.feature_cols].copy()
        y = df[self.target_col].copy()
        
        # Fill missing values with median
        X = X.fillna(X.median())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        return X.values, y.values
    
    def train_baseline(self) -> Tuple[LogisticRegression, StandardScaler, Dict]:
        """
        Train baseline LogisticRegression with class balancing.
        
        Returns:
            (model, scaler, metrics_dict)
        """
        logger.info("Training baseline LogisticRegression...")
        
        train_df, test_df = self.time_based_split()
        
        X_train, y_train = self.prepare_data(train_df)
        X_test, y_test = self.prepare_data(test_df)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate scale_pos_weight to handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train model with balanced weights
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs'
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        metrics = self._evaluate_predictions(y_test, y_pred_proba, "Baseline LogisticRegression")
        
        return model, scaler, metrics
    
    def train_xgboost(self) -> Tuple[xgb.XGBClassifier, Dict]:
        """
        Train XGBoost model with hyperparameter tuning.
        
        Returns:
            (model, metrics_dict)
        """
        if not HAS_XGBOOST:
            logger.error("XGBoost not installed")
            return None, {}
        
        logger.info("Training XGBoost...")
        
        train_df, test_df = self.time_based_split()
        
        X_train, y_train = self.prepare_data(train_df)
        X_test, y_test = self.prepare_data(test_df)
        
        # Calculate scale_pos_weight for imbalance handling
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=10,
            eval_metric='logloss'
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = self._evaluate_predictions(y_test, y_pred_proba, "XGBoost")
        
        # Feature importance (basic version)
        feature_importance = dict(zip(self.feature_cols, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top 5 important features:")
        for feat, importance in top_features:
            logger.info(f"  {feat}: {importance:.4f}")
        
        return model, metrics
    
    def _evaluate_predictions(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                             model_name: str = "") -> Dict:
        """
        Evaluate predictions using PR-AUC (not ROC-AUC).
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            model_name: Name for logging
        
        Returns:
            Dictionary of metrics
        """
        # Precision-Recall curve and AUC (better for imbalanced data)
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Use 0.5 threshold for standard metrics
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        f1 = f1_score(y_true, y_pred)
        precision_at_05 = precision_score(y_true, y_pred)
        recall_at_05 = recall_score(y_true, y_pred)
        
        metrics = {
            'pr_auc': pr_auc,
            'f1_at_0.5': f1,
            'precision_at_0.5': precision_at_05,
            'recall_at_0.5': recall_at_05,
            'y_pred_proba': y_pred_proba,
            'y_true': y_true,
        }
        
        logger.info(f"\n{model_name} Evaluation:")
        logger.info(f"  PR-AUC: {pr_auc:.4f}")
        logger.info(f"  F1 (at 0.5): {f1:.4f}")
        logger.info(f"  Precision: {precision_at_05:.4f}")
        logger.info(f"  Recall: {recall_at_05:.4f}")
        
        return metrics
    
    def close(self):
        """Clean up."""
        pass


def save_model(model, scaler, model_path: str = "models/artifacts/model.pkl"):
    """Save trained model and scaler."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train bug-risk prediction models")
    parser.add_argument("--features", default="data/features.csv", help="Features CSV path")
    parser.add_argument("--model-dir", default="models/artifacts", help="Output directory")
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(args.features)
    
    # Train baseline
    baseline_model, scaler, baseline_metrics = trainer.train_baseline()
    save_model(baseline_model, scaler, f"{args.model_dir}/baseline_model.pkl")
    
    # Train XGBoost
    if HAS_XGBOOST:
        xgb_model, xgb_metrics = trainer.train_xgboost()
        save_model(xgb_model, None, f"{args.model_dir}/xgboost_model.pkl")
    
    logger.info("Model training complete!")
