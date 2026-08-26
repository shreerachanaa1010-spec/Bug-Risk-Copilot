"""
Phase 3.4: SHAP Explainability Analysis
Generates feature importance and individual prediction explanations.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Generate SHAP explanations for model predictions."""
    
    def __init__(self, model_path: str, features_path: str):
        """
        Initialize SHAP explainer.
        
        Args:
            model_path: Path to saved XGBoost model
            features_path: Path to features CSV
        """
        if not HAS_SHAP:
            raise ImportError("shap required. Install: pip install shap")
        
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load features
        self.features_df = pd.read_csv(features_path)
        self.feature_cols = [col for col in self.features_df.columns
                            if col not in ['commit_hash', 'author', 'author_date', 
                                         'day_of_week', 'is_bug_introducing']]
        
        logger.info(f"Loaded model and features ({len(self.feature_cols)} features)")
    
    def create_explainer(self, sample_size: int = 100):
        """
        Create a SHAP TreeExplainer (works on tree-based models).
        
        Args:
            sample_size: Number of samples to use for background
        
        Returns:
            shap.TreeExplainer instance
        """
        X = self.features_df[self.feature_cols].fillna(0).values
        
        # Use subset of data for faster computation
        if len(X) > sample_size:
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X
        
        logger.info(f"Creating SHAP explainer with {len(X_sample)} background samples")
        explainer = shap.TreeExplainer(self.model['model'])
        
        return explainer
    
    def global_feature_importance(self, explainer: shap.TreeExplainer) -> Dict:
        """
        Compute global feature importance from SHAP values.
        
        Returns:
            Dictionary of feature names to importance scores
        """
        X = self.features_df[self.feature_cols].fillna(0).values
        
        logger.info("Computing global SHAP values...")
        shap_values = explainer.shap_values(X)
        
        # For binary classification, take positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        feature_importance = dict(zip(self.feature_cols, importance))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("\nGlobal Feature Importance (SHAP):")
        logger.info("-" * 50)
        for feat, imp in sorted_features[:10]:
            logger.info(f"{feat:.<40} {imp:.4f}")
        
        return feature_importance
    
    def explain_prediction(self, explainer: shap.TreeExplainer, 
                          commit_hash: str, top_k: int = 5) -> Dict:
        """
        Explain a single prediction with SHAP.
        
        Args:
            explainer: SHAP explainer
            commit_hash: Hash of commit to explain
            top_k: Number of top features to return
        
        Returns:
            Dictionary with explanation data
        """
        # Find commit in features
        commit_idx = self.features_df[self.features_df['commit_hash'] == commit_hash].index
        if len(commit_idx) == 0:
            logger.warning(f"Commit {commit_hash} not found")
            return {}
        
        idx = commit_idx[0]
        X = self.features_df[self.feature_cols].fillna(0).values
        
        # Get SHAP values for this instance
        shap_value = explainer.shap_values(X[idx:idx+1])
        
        if isinstance(shap_value, list):
            shap_value = shap_value[1][0]  # positive class
        else:
            shap_value = shap_value[0]
        
        # Get prediction
        prediction = self.model['model'].predict_proba(X[idx:idx+1])[0, 1]
        
        # Top contributing features
        top_indices = np.argsort(np.abs(shap_value))[-top_k:][::-1]
        
        features_contribution = []
        for i in top_indices:
            features_contribution.append({
                'feature': self.feature_cols[i],
                'value': X[idx, i],
                'shap_value': float(shap_value[i]),
            })
        
        explanation = {
            'commit_hash': commit_hash,
            'prediction_score': float(prediction),
            'top_features': features_contribution,
        }
        
        logger.info(f"\nExplanation for {commit_hash}:")
        logger.info(f"  Risk score: {prediction:.4f}")
        logger.info(f"  Top contributing features:")
        for feat_info in features_contribution:
            logger.info(f"    {feat_info['feature']}: {feat_info['shap_value']:.4f} "
                       f"(value: {feat_info['value']:.2f})")
        
        return explanation
    
    def batch_explanations(self, explainer: shap.TreeExplainer, 
                          commit_hashes: list, top_k: int = 5) -> list:
        """Generate explanations for multiple commits."""
        explanations = []
        for commit_hash in commit_hashes:
            explanation = self.explain_prediction(explainer, commit_hash, top_k)
            if explanation:
                explanations.append(explanation)
        return explanations


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SHAP explanations")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--features", default="data/features.csv", help="Features CSV path")
    parser.add_argument("--commit", help="Specific commit hash to explain")
    
    args = parser.parse_args()
    
    explainer_obj = SHAPExplainer(args.model, args.features)
    explainer = explainer_obj.create_explainer()
    
    # Global importance
    importance = explainer_obj.global_feature_importance(explainer)
    
    # Specific commit explanation
    if args.commit:
        explainer_obj.explain_prediction(explainer, args.commit)
    
    logger.info("SHAP analysis complete!")
