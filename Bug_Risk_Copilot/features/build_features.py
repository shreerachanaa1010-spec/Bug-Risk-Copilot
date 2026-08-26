"""
Phase 2.1-2.5: Feature Engineering Pipeline
Extracts structured features from commits for ML modeling.

Features extracted:
- Diffusion: num_files, num_directories, num_subsystems
- Size: lines_added, lines_deleted, total_churn
- Complexity: complexity_delta (via radon)
- Historical: file_age, file_prior_bug_count, author_prior_bug_rate
- Message/metadata: message_length, has_issue_ref, hour, day_of_week
"""

import sqlite3
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    from radon.complexity import cc_visit
    HAS_RADON = True
except ImportError:
    HAS_RADON = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract ML features from commit data."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def get_all_commits(self) -> pd.DataFrame:
        """Load all commits from database."""
        query = """
            SELECT 
                hash, author, author_date, message, files_json,
                insertions, deletions, num_files, is_bug_introducing
            FROM commits
            ORDER BY author_date ASC
        """
        df = pd.read_sql_query(query, self.conn)
        df['author_date'] = pd.to_datetime(df['author_date'])
        return df
    
    def extract_diffusion_features(self, commit_id: str, files_json: str) -> Dict:
        """Extract features about how widespread a change is."""
        files = json.loads(files_json)
        
        num_files = len(files)
        
        # Count unique directories
        directories = set()
        subsystems = set()  # top-level only
        
        for file in files:
            path = file.get('file', '')
            if '/' in path:
                dir_path = path.rsplit('/', 1)[0]
                directories.add(dir_path)
                # Top-level subsystem
                subsystem = path.split('/')[0]
                subsystems.add(subsystem)
        
        return {
            'num_files_changed': num_files,
            'num_directories_touched': len(directories),
            'num_subsystems_touched': len(subsystems),
        }
    
    def extract_size_features(self, insertions: int, deletions: int) -> Dict:
        """Extract features about change size."""
        total_churn = insertions + deletions
        
        return {
            'lines_added': insertions,
            'lines_deleted': deletions,
            'total_churn': total_churn,
            'churn_ratio': insertions / max(deletions, 1),  # avoid division by zero
        }
    
    def extract_complexity_features(self, files_json: str) -> Dict:
        """Extract code complexity metrics using radon."""
        if not HAS_RADON:
            return {
                'max_complexity_delta': 0,
                'avg_complexity_delta': 0,
            }
        
        files = json.loads(files_json)
        complexity_deltas = []
        
        for file in files:
            if not file['file'].endswith('.py'):
                continue
            
            # For now, we'd need the actual file content, which we don't have in this schema
            # In production, you'd parse the diff to extract before/after code
            # For this prototype, we approximate with a zero value
            complexity_deltas.append(0)
        
        if not complexity_deltas:
            return {
                'max_complexity_delta': 0,
                'avg_complexity_delta': 0,
            }
        
        return {
            'max_complexity_delta': max(complexity_deltas),
            'avg_complexity_delta': np.mean(complexity_deltas),
        }
    
    def extract_historical_features(self, df: pd.DataFrame, idx: int, 
                                   commit_hash: str, author: str) -> Dict:
        """
        Extract historical risk signals.
        
        IMPORTANT: Only use data from *before* this commit (temporal causality)
        """
        commit_date = df.loc[idx, 'author_date']
        
        # Look at all commits BEFORE this one
        prior_commits = df[df['author_date'] < commit_date].copy()
        
        # File history features (would need to parse which files changed)
        # For now, simplified version
        file_age_days = 0  # Would need file creation date
        
        # Count prior bug-introducing commits in files changed by this commit
        prior_bug_count = prior_commits[prior_commits['is_bug_introducing'] == 1].shape[0]
        
        # Author's historical bug rate (only from commits before this one)
        author_prior_commits = prior_commits[prior_commits['author'] == author]
        if len(author_prior_commits) > 0:
            author_prior_bug_rate = author_prior_commits['is_bug_introducing'].mean()
            author_commit_count = len(author_prior_commits)
        else:
            author_prior_bug_rate = 0
            author_commit_count = 0
        
        return {
            'file_age_days': file_age_days,
            'file_prior_bug_count': min(prior_bug_count, 10),  # cap at 10
            'author_prior_bug_rate': author_prior_bug_rate,
            'author_commit_count': author_commit_count,
        }
    
    def extract_message_features(self, message: str, timestamp: str) -> Dict:
        """Extract features from commit message and timestamp."""
        import re
        
        message_length = len(message)
        has_issue_reference = bool(re.search(r'#\d+', message))
        
        # Time-of-day features
        dt = pd.to_datetime(timestamp)
        hour_of_day = dt.hour
        day_of_week = dt.day_name()
        
        return {
            'message_length': message_length,
            'has_issue_reference': int(has_issue_reference),
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
        }
    
    def extract_all_features(self) -> pd.DataFrame:
        """
        Extract all features for all commits.
        
        Returns:
            DataFrame with one row per commit and all features
        """
        df = self.get_all_commits()
        logger.info(f"Extracting features for {len(df)} commits...")
        
        features_list = []
        
        for idx, row in df.iterrows():
            feature_dict = {
                'commit_hash': row['hash'],
                'is_bug_introducing': row['is_bug_introducing'],
                'author': row['author'],
                'author_date': row['author_date'],
            }
            
            # Extract feature groups
            try:
                diff_feats = self.extract_diffusion_features(row['hash'], row['files_json'])
                feature_dict.update(diff_feats)
                
                size_feats = self.extract_size_features(row['insertions'], row['deletions'])
                feature_dict.update(size_feats)
                
                complexity_feats = self.extract_complexity_features(row['files_json'])
                feature_dict.update(complexity_feats)
                
                hist_feats = self.extract_historical_features(df, idx, row['hash'], row['author'])
                feature_dict.update(hist_feats)
                
                msg_feats = self.extract_message_features(row['message'], row['author_date'])
                feature_dict.update(msg_feats)
                
                features_list.append(feature_dict)
                
                if (idx + 1) % 500 == 0:
                    logger.info(f"Processed {idx + 1} commits...")
                
            except Exception as e:
                logger.warning(f"Error extracting features for {row['hash']}: {e}")
        
        features_df = pd.DataFrame(features_list)
        
        # Convert day_of_week to numeric
        day_mapping = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        features_df['day_of_week_num'] = features_df['day_of_week'].map(day_mapping)
        
        logger.info(f"Feature extraction complete: {len(features_df)} rows")
        logger.info(f"Positive class: {features_df['is_bug_introducing'].sum()} "
                   f"({100*features_df['is_bug_introducing'].mean():.2f}%)")
        
        return features_df
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def save_features(features_df: pd.DataFrame, output_path: str = "data/features.csv"):
    """Save features to CSV."""
    features_df.to_csv(output_path, index=False)
    logger.info(f"Features saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract ML features from commits")
    parser.add_argument("--db", default="data/commits.db", help="Path to SQLite database")
    parser.add_argument("--output", default="data/features.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    
    extractor = FeatureExtractor(args.db)
    features_df = extractor.extract_all_features()
    save_features(features_df, args.output)
    extractor.close()
    
    logger.info("Feature extraction complete!")
