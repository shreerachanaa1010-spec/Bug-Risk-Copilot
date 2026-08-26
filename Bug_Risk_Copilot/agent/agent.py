"""
Phase 4: LangGraph Agent for Bug-Risk Investigation and Explanation
Orchestrates multi-step reasoning:
  1. fetch_context: Retrieve model prediction and SHAP features
  2. investigate: Query FAISS for similar historical bug commits
  3. explain: Call LLM to generate natural language explanation
  4. decide_action: Determine routing (comment vs escalate)
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json

try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage
    HAS_LANGCHAIN_GOOGLE = True
except ImportError:
    HAS_LANGCHAIN_GOOGLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define agent state schema
class AgentState(dict):
    """State schema for the bug-risk agent."""
    
    # Input
    commit_hash: str
    diff_text: str
    
    # From ML model
    risk_score: float
    top_features: List[Dict]  # SHAP top features
    
    # From investigation
    similar_commits: List[Dict]
    needs_more_context: bool
    investigation_round: int
    
    # Output
    explanation: str
    action: str  # "comment" or "escalate"
    confidence: float


class BugRiskAgent:
    """LangGraph-based agent for bug-risk analysis."""
    
    def __init__(self, threshold: float = 0.6, gemini_api_key: Optional[str] = None):
        """
        Initialize agent.
        
        Args:
            threshold: Risk score threshold (above = escalate, below = comment)
            gemini_api_key: Optional Google Gemini API key
        """
        if not HAS_LANGGRAPH:
            raise ImportError("langgraph required. Install: pip install langgraph")
        
        self.threshold = threshold
        self.gemini_api_key = gemini_api_key
        self.max_investigation_rounds = 2
        
        # Optional: Initialize LLM if API key available
        if gemini_api_key and HAS_LANGCHAIN_GOOGLE:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=gemini_api_key,
                temperature=0.7
            )
        else:
            self.llm = None
            logger.warning("No Gemini API key; LLM explanations will be mocked")
    
    def fetch_context(self, state: dict) -> dict:
        """
        Node 1: Fetch prediction context.
        
        In production, this would call the ML model if not already populated.
        For now, assumes state contains pre-computed model outputs.
        """
        logger.info(f"[fetch_context] Retrieving context for {state.get('commit_hash', 'unknown')}")
        
        # Validate required fields
        if 'risk_score' not in state:
            state['risk_score'] = 0.5
        if 'top_features' not in state:
            state['top_features'] = []
        if 'needs_more_context' not in state:
            state['needs_more_context'] = True
        if 'investigation_round' not in state:
            state['investigation_round'] = 0
        if 'similar_commits' not in state:
            state['similar_commits'] = []
        
        logger.info(f"  Risk score: {state['risk_score']:.3f}")
        logger.info(f"  Top features: {[f['feature'] for f in state['top_features'][:3]]}")
        
        return state
    
    def investigate(self, state: dict) -> dict:
        """
        Node 2: Investigate similar historical bug commits.
        
        Uses FAISS embeddings to find semantically similar past bugs.
        Can loop if findings are ambiguous.
        """
        logger.info(f"[investigate] Round {state['investigation_round'] + 1}")
        
        diff_text = state.get('diff_text', '')
        
        # In production, this would query FAISS index
        # For MVP, we mock a retrieval
        state['similar_commits'] = self._mock_faiss_search(diff_text)
        
        logger.info(f"  Found {len(state['similar_commits'])} similar commits")
        
        # Decide if we need more context
        # (In this simple version, we don't loop)
        state['needs_more_context'] = False
        state['investigation_round'] += 1
        
        return state
    
    def explain(self, state: dict) -> dict:
        """
        Node 3: Generate natural language explanation via LLM.
        """
        logger.info("[explain] Generating explanation...")
        
        risk_score = state['risk_score']
        top_features = state.get('top_features', [])
        similar_commits = state.get('similar_commits', [])
        
        # Build prompt
        prompt = self._build_explanation_prompt(
            risk_score, top_features, similar_commits
        )
        
        if self.llm:
            try:
                # Call LLM
                response = self.llm.invoke([HumanMessage(content=prompt)])
                explanation = response.content
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                explanation = self._mock_explanation(risk_score, top_features)
        else:
            explanation = self._mock_explanation(risk_score, top_features)
        
        state['explanation'] = explanation
        logger.info(f"  Explanation: {explanation[:200]}...")
        
        return state
    
    def decide_action(self, state: dict) -> dict:
        """
        Node 4: Decide routing based on risk score.
        
        The LLM never overrides the model's decision, only explains it.
        """
        logger.info("[decide_action] Determining action...")
        
        risk_score = state['risk_score']
        
        if risk_score >= self.threshold:
            state['action'] = 'escalate'
            state['confidence'] = min(risk_score, 0.99)
        else:
            state['action'] = 'comment'
            state['confidence'] = 1 - risk_score
        
        logger.info(f"  Action: {state['action']} (confidence: {state['confidence']:.3f})")
        
        return state
    
    def _mock_faiss_search(self, diff_text: str, k: int = 3) -> List[Dict]:
        """Mock FAISS search (for MVP without real embeddings)."""
        # In production, this queries the FAISS index
        return [
            {
                'commit_hash': 'abc123',
                'message': 'Fix authentication bypass in session handler',
                'date': '2025-01-15',
                'files': ['auth/session.py'],
                'similarity_score': 0.87,
            },
            {
                'commit_hash': 'def456',
                'message': 'Fix race condition in cache invalidation',
                'date': '2025-01-10',
                'files': ['cache/redis_client.py'],
                'similarity_score': 0.72,
            },
        ]
    
    def _build_explanation_prompt(self, risk_score: float, 
                                 top_features: List[Dict],
                                 similar_commits: List[Dict]) -> str:
        """Construct prompt for LLM explanation."""
        features_text = "\n".join([
            f"- {f['feature']}: {f['value']:.2f} (impact: {f.get('shap_value', 0):.3f})"
            for f in top_features[:3]
        ])
        
        similar_text = "\n".join([
            f"- {c['message']} (similarity: {c['similarity_score']:.2f})"
            for c in similar_commits[:2]
        ])
        
        prompt = f"""You are a code review assistant analyzing a potential bug-introducing commit.

Risk Score (0-1): {risk_score:.3f}

Top Contributing Factors:
{features_text}

Similar Historical Bug Commits:
{similar_text}

Provide a 2-3 sentence explanation of the risk, mentioning specific factors and historical precedents.
Be concise and actionable."""
        
        return prompt
    
    def _mock_explanation(self, risk_score: float, top_features: List[Dict]) -> str:
        """Generate mock explanation (for MVP without LLM)."""
        if risk_score > 0.7:
            severity = "HIGH"
        elif risk_score > 0.5:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        top_feature = top_features[0]['feature'] if top_features else "unknown"
        
        return (f"Risk Level: {severity} ({risk_score:.2f}). This commit's primary risk factor is "
               f"{top_feature}. Recommend review before merge.")
    
    def build_graph(self):
        """
        Build and compile the LangGraph agent graph.
        
        Graph flow:
          fetch_context → investigate → explain → decide_action
        """
        from langgraph.graph import StateGraph, END
        
        graph = StateGraph(dict)
        
        # Add nodes
        graph.add_node("fetch_context", self.fetch_context)
        graph.add_node("investigate", self.investigate)
        graph.add_node("explain", self.explain)
        graph.add_node("decide_action", self.decide_action)
        
        # Add edges
        graph.set_entry_point("fetch_context")
        graph.add_edge("fetch_context", "investigate")
        graph.add_edge("investigate", "explain")
        graph.add_edge("explain", "decide_action")
        graph.add_edge("decide_action", END)
        
        # Compile graph
        self.app = graph.compile()
        logger.info("LangGraph agent compiled successfully")
        
        return self.app
    
    def run(self, state: dict) -> dict:
        """
        Run the agent on a state.
        
        Args:
            state: Dictionary with commit_hash, diff_text, risk_score, top_features
        
        Returns:
            Updated state with explanation and action
        """
        if not hasattr(self, 'app'):
            self.build_graph()
        
        # Run the graph
        output_state = self.app.invoke(state)
        
        return output_state


# Example usage
if __name__ == "__main__":
    # Create agent (without Gemini API key for MVP)
    agent = BugRiskAgent(threshold=0.6)
    agent.build_graph()
    
    # Example input state
    example_state = {
        'commit_hash': 'abc1234567',
        'diff_text': 'Changes to authentication module...',
        'risk_score': 0.75,
        'top_features': [
            {'feature': 'total_churn', 'value': 245.0, 'shap_value': 0.15},
            {'feature': 'file_prior_bug_count', 'value': 3.0, 'shap_value': 0.12},
            {'feature': 'num_files_changed', 'value': 8.0, 'shap_value': 0.08},
        ],
    }
    
    # Run agent
    logger.info("Running agent on example state...")
    result = agent.run(example_state)
    
    # Print results
    logger.info("\n=== Agent Output ===")
    logger.info(f"Action: {result.get('action')}")
    logger.info(f"Explanation: {result.get('explanation')}")
    logger.info(f"Confidence: {result.get('confidence', 0):.3f}")
