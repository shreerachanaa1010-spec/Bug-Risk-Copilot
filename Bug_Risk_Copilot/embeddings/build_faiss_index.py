"""
Phase 2.6: Build FAISS Index for Historical Bug Commits
Embeds all bug-introducing commits and builds a vector search index.

This enables the agent to retrieve similar historical bug commits for context.
"""

import sqlite3
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    logger = logging.getLogger(__name__)
    logger.warning("sentence-transformers or faiss not installed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSIndexBuilder:
    """Build and manage FAISS vector search index for bug commits."""
    
    def __init__(self, db_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name
        
        if not HAS_EMBEDDINGS:
            raise ImportError("sentence-transformers and faiss-cpu required. "
                            "Install: pip install sentence-transformers faiss-cpu")
        
        # Load embedding model (384-dim, fast, no API calls)
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Will store metadata about indexed commits
        self.metadata = []  # List of (commit_hash, file_path, date, diff_snippet)
    
    def get_bug_introducing_commits(self) -> pd.DataFrame:
        """Get all commits labeled as bug-introducing."""
        query = """
            SELECT hash, message, files_json, insertions, deletions, author_date
            FROM commits 
            WHERE is_bug_introducing = 1
            ORDER BY author_date DESC
        """
        df = pd.read_sql_query(query, self.conn)
        logger.info(f"Found {len(df)} bug-introducing commits")
        return df
    
    def create_embedding_text(self, message: str, files_json: str) -> str:
        """
        Create a text representation of a commit for embedding.
        
        Combines commit message + file list for better semantic matching.
        """
        files = json.loads(files_json)
        file_names = ', '.join([f['file'] for f in files[:10]])  # Top 10 files
        
        text = f"{message}\n\nFiles changed: {file_names}"
        return text
    
    def build_index(self, output_dir: str = "models/artifacts") -> faiss.Index:
        """
        Build FAISS index from bug-introducing commits.
        
        Returns:
            FAISS index object
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get bug commits
        bug_commits = self.get_bug_introducing_commits()
        
        if len(bug_commits) == 0:
            logger.error("No bug-introducing commits found!")
            return None
        
        # Create embeddings
        logger.info(f"Embedding {len(bug_commits)} bug commits...")
        embeddings = []
        
        for idx, row in bug_commits.iterrows():
            text = self.create_embedding_text(row['message'], row['files_json'])
            
            # Embed the text
            embedding = self.model.encode(text, convert_to_numpy=True)
            embeddings.append(embedding)
            
            # Store metadata
            self.metadata.append({
                'commit_hash': row['hash'],
                'message': row['message'][:200],  # truncate for storage
                'author_date': row['author_date'],
                'files_count': len(json.loads(row['files_json'])),
                'insertions': row['insertions'],
                'deletions': row['deletions'],
            })
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Embedded {idx + 1} commits...")
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        logger.info(f"Embeddings shape: {embeddings_array.shape}")
        
        # Create FAISS index (L2 distance, simple flat index for now)
        # For production with 100k+ commits, use IndexIVFFlat or IndexHNSW
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings_array)
        
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        
        # Save index and metadata
        index_path = str(Path(output_dir) / "bug_commits.index")
        metadata_path = str(Path(output_dir) / "bug_commits_metadata.pkl")
        
        faiss.write_index(index, index_path)
        logger.info(f"Index saved to {index_path}")
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Metadata saved to {metadata_path}")
        
        return index
    
    def search_similar(self, query_text: str, index: faiss.Index, 
                      k: int = 3) -> List[Dict]:
        """
        Search for similar bug commits.
        
        Args:
            query_text: Commit message or diff to search for
            index: FAISS index
            k: Number of results to return
        
        Returns:
            List of similar commit metadata dicts
        """
        # Embed query
        query_embedding = self.model.encode(query_text, convert_to_numpy=True)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        distances, indices = index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['distance'] = float(distances[0][i])
                results.append(result)
        
        return results
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def load_index(index_path: str, metadata_path: str) -> Tuple[faiss.Index, List[Dict]]:
    """Load a pre-built FAISS index."""
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index for bug commits")
    parser.add_argument("--db", default="data/commits.db", help="Path to SQLite database")
    parser.add_argument("--output", default="models/artifacts", help="Output directory")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", 
                       help="Sentence transformer model name")
    
    args = parser.parse_args()
    
    builder = FAISSIndexBuilder(args.db, args.model)
    index = builder.build_index(args.output)
    builder.close()
    
    if index:
        logger.info("FAISS index built successfully!")
    else:
        logger.error("Failed to build FAISS index")
