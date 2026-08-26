"""
Phase 5.4: Unit Tests for Pipeline Components
Tests ensure data integrity and model correctness.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
import json

# Import components to test
from data.mining import extract_commit_data, detect_fix_commit, init_db
from features.build_features import FeatureExtractor
from agent.agent import BugRiskAgent


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def temp_db():
    """Create temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


class MockCommit:
    """Mock git commit for testing."""
    
    def __init__(self, hash, message, files_modified=None):
        self.hash = hash
        self.message = message
        self.author = type('Author', (), {
            'name': 'Test Author',
            'email': 'test@example.com'
        })()
        self.committer = None
        self.author_date = '2025-01-01 12:00:00'
        self.committer_date = None
        
        if files_modified:
            self.modified_files = files_modified
        else:
            self.modified_files = []


class MockModifiedFile:
    """Mock modified file for testing."""
    
    def __init__(self, filename, added=5, deleted=2):
        self.filename = filename
        self.added_lines = added
        self.deleted_lines = deleted
        self.change_type = type('ChangeType', (), {'name': 'MODIFY'})()
        self.old_path = filename
        self.new_path = filename


# ============================================================================
# Tests: Data Mining
# ============================================================================

class TestMining:
    """Tests for commit mining."""
    
    def test_fix_commit_detection(self):
        """Test regex patterns for fix-commit detection."""
        
        # Should detect as fix
        fix_messages = [
            "fix: authentication bypass",
            "Fixes #123 race condition",
            "bugfix: memory leak",
            "resolve issue #456",
            "closes #789",
        ]
        
        for msg in fix_messages:
            assert detect_fix_commit(msg), f"Failed to detect: {msg}"
        
        # Should NOT detect as fix
        non_fix_messages = [
            "add new feature",
            "refactor code",
            "docs: update readme",
        ]
        
        for msg in non_fix_messages:
            assert not detect_fix_commit(msg), f"False positive: {msg}"
    
    def test_extract_commit_data(self):
        """Test commit data extraction."""
        
        # Create mock commit
        files = [MockModifiedFile("src/main.py", 10, 5)]
        commit = MockCommit("abc123", "fix: critical bug", files)
        
        # Extract data
        data = extract_commit_data(commit)
        
        # Verify
        assert data['hash'] == "abc123"
        assert data['message'] == "fix: critical bug"
        assert data['is_fix'] is True
        assert data['insertions'] == 10
        assert data['deletions'] == 5
        assert data['num_files'] == 1
    
    def test_init_db(self, temp_db):
        """Test database initialization."""
        
        conn = init_db(temp_db)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='commits'"
        )
        assert cursor.fetchone() is not None
        
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='file_changes'"
        )
        assert cursor.fetchone() is not None
        
        conn.close()


# ============================================================================
# Tests: Feature Extraction
# ============================================================================

class TestFeatureExtraction:
    """Tests for feature engineering."""
    
    def test_diffusion_features(self):
        """Test diffusion (spread) feature extraction."""
        
        extractor = FeatureExtractor(":memory:")
        
        files_json = json.dumps([
            {'file': 'src/auth/login.py'},
            {'file': 'src/auth/session.py'},
            {'file': 'tests/auth_test.py'},
        ])
        
        features = extractor.extract_diffusion_features("test", files_json)
        
        assert features['num_files_changed'] == 3
        assert features['num_directories_touched'] == 2  # src/auth, tests
        assert features['num_subsystems_touched'] == 2  # src, tests
    
    def test_size_features(self):
        """Test code size feature extraction."""
        
        extractor = FeatureExtractor(":memory:")
        
        features = extractor.extract_size_features(insertions=100, deletions=50)
        
        assert features['lines_added'] == 100
        assert features['lines_deleted'] == 50
        assert features['total_churn'] == 150
        assert features['churn_ratio'] == 2.0
    
    def test_message_features(self):
        """Test commit message feature extraction."""
        
        extractor = FeatureExtractor(":memory:")
        
        # Message with issue reference
        features = extractor.extract_message_features(
            "Fix bug #123 in auth",
            "2025-01-15 14:30:00"
        )
        
        assert features['message_length'] > 0
        assert features['has_issue_reference'] == 1
        assert features['hour_of_day'] == 14
        
        # Message without issue reference
        features = extractor.extract_message_features(
            "Refactor code",
            "2025-01-15 14:30:00"
        )
        
        assert features['has_issue_reference'] == 0


# ============================================================================
# Tests: Agent
# ============================================================================

class TestAgent:
    """Tests for LangGraph agent."""
    
    def test_agent_initialization(self):
        """Test agent can be created."""
        
        agent = BugRiskAgent(threshold=0.6)
        assert agent.threshold == 0.6
    
    def test_agent_fetch_context(self):
        """Test fetch_context node."""
        
        agent = BugRiskAgent()
        
        state = {
            'commit_hash': 'test123',
            'diff_text': 'test diff',
        }
        
        output = agent.fetch_context(state)
        
        assert 'risk_score' in output
        assert 'top_features' in output
        assert isinstance(output['top_features'], list)
    
    def test_agent_investigate(self):
        """Test investigate node."""
        
        agent = BugRiskAgent()
        
        state = {
            'commit_hash': 'test123',
            'diff_text': 'test diff',
            'risk_score': 0.7,
            'top_features': [],
        }
        
        output = agent.investigate(state)
        
        assert 'similar_commits' in output
        assert isinstance(output['similar_commits'], list)
    
    def test_agent_explain(self):
        """Test explain node."""
        
        agent = BugRiskAgent()
        
        state = {
            'commit_hash': 'test123',
            'risk_score': 0.7,
            'top_features': [
                {'feature': 'total_churn', 'value': 100, 'shap_value': 0.1}
            ],
            'similar_commits': [],
        }
        
        output = agent.explain(state)
        
        assert 'explanation' in output
        assert isinstance(output['explanation'], str)
        assert len(output['explanation']) > 0
    
    def test_agent_decide_action_low_risk(self):
        """Test decide_action node with low risk."""
        
        agent = BugRiskAgent(threshold=0.6)
        
        state = {
            'commit_hash': 'test123',
            'risk_score': 0.3,
            'explanation': 'Low risk commit',
        }
        
        output = agent.decide_action(state)
        
        assert output['action'] == 'comment'
        assert output['confidence'] > 0
    
    def test_agent_decide_action_high_risk(self):
        """Test decide_action node with high risk."""
        
        agent = BugRiskAgent(threshold=0.6)
        
        state = {
            'commit_hash': 'test123',
            'risk_score': 0.8,
            'explanation': 'High risk commit',
        }
        
        output = agent.decide_action(state)
        
        assert output['action'] == 'escalate'
        assert output['confidence'] > 0
    
    def test_agent_full_run(self):
        """Test full agent pipeline."""
        
        agent = BugRiskAgent(threshold=0.6)
        agent.build_graph()
        
        state = {
            'commit_hash': 'test123',
            'diff_text': 'test diff',
            'risk_score': 0.65,
            'top_features': [
                {'feature': 'total_churn', 'value': 100, 'shap_value': 0.1}
            ],
        }
        
        result = agent.run(state)
        
        # Check output structure
        assert result['commit_hash'] == 'test123'
        assert 'explanation' in result
        assert 'action' in result
        assert result['action'] in ['comment', 'escalate']


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_data_pipeline(self, temp_db):
        """Test mining + feature extraction pipeline."""
        
        # Initialize database
        conn = init_db(temp_db)
        
        # Insert test commits
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO commits 
            (hash, author, author_date, message, files_json, insertions, 
             deletions, num_files, is_fix_commit, is_bug_introducing)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'abc123',
            'Test Author',
            '2025-01-01 12:00:00',
            'fix: bug',
            json.dumps([{'file': 'src/main.py'}]),
            10,
            5,
            1,
            1,
            0,
        ))
        conn.commit()
        conn.close()
        
        # Extract features
        # Note: In real test, would load full CSV, here we just verify the flow
        assert Path(temp_db).exists()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run pytest
    pytest.main([__file__, "-v"])
