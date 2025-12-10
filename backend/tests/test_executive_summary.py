"""
Tests for executive summary functionality with evidence integration.

Tests the modified summarize_input function and pipeline integration to ensure
summaries are generated after verification with evidence and reasoning included.
"""

import pytest
from unittest.mock import Mock, patch
from pipeline import check_text, summarize_input


class TestExecutiveSummary:
    """Test executive summary generation with evidence integration."""

    @patch('pipeline.summary.chat')
    def test_summarize_input_without_evidence(self, mock_chat):
        """Test backward compatibility - summarize_input works without evidence_results."""
        mock_chat.return_value = "This is a test summary of the input text."

        result = summarize_input("Test input text")

        assert isinstance(result, str)
        assert len(result) > 0
        mock_chat.assert_called_once()

    @patch('pipeline.summary.chat')
    def test_summarize_input_with_evidence(self, mock_chat):
        """Test summarize_input with evidence results included."""
        mock_chat.return_value = "Comprehensive summary including verification results."

        evidence_results = [{
            'claim': 'Test claim about AI',
            'verdict': 'TRUE',
            'confidence': 0.95,
            'reasoning': 'Strong evidence from multiple sources supports this claim.',
            'sources': [
                {'title': 'AI News Today', 'url': 'https://ai-news.com'},
                {'title': 'Tech Review', 'url': 'https://tech-review.com'}
            ]
        }]

        result = summarize_input("Original AI claim text", evidence_results=evidence_results)

        assert isinstance(result, str)
        assert len(result) > 0

        # Check that the call included evidence context
        call_args = mock_chat.call_args
        system_prompt = call_args[0][0]
        user_content = call_args[0][1]

        assert "Verification Results:" in user_content
        assert "Test claim about AI" in user_content
        assert "Verdict: TRUE" in user_content
        assert "AI News Today" in user_content

    @patch('pipeline.summary.chat')
    def test_summarize_input_multiple_claims(self, mock_chat):
        """Test summarize_input with multiple evidence results."""
        mock_chat.return_value = "Summary covering multiple verified claims."

        evidence_results = [
            {
                'claim': 'Claim 1: AI is advancing rapidly',
                'verdict': 'TRUE',
                'confidence': 0.9,
                'reasoning': 'Multiple studies confirm this trend.',
                'sources': [{'title': 'Study 1', 'url': 'https://study1.com'}]
            },
            {
                'claim': 'Claim 2: AI will create jobs',
                'verdict': 'MOSTLY_TRUE',
                'confidence': 0.7,
                'reasoning': 'Evidence suggests net positive job impact.',
                'sources': [{'title': 'Report 1', 'url': 'https://report1.com'}]
            }
        ]

        result = summarize_input("Original text with multiple claims", evidence_results=evidence_results)

        assert isinstance(result, str)
        assert len(result) > 0

        # Verify both claims are included in context
        user_content = mock_chat.call_args[0][1]
        assert "Claim 1:" in user_content
        assert "Claim 2:" in user_content
        assert "Verdict: TRUE" in user_content
        assert "Verdict: MOSTLY_TRUE" in user_content


class TestPipelineIntegration:
    """Test that pipeline generates summaries after verification."""

    @patch('pipeline.summary.chat')
    @patch('search.base.collect_evidence')
    def test_single_claim_pipeline_summary_timing(self, mock_collect_evidence, mock_chat):
        """Test that summary is generated after verification for single claims."""
        # Mock the LLM calls with proper string responses
        mock_chat.side_effect = [
            # detect_intent response - JSON string
            '{"intent": "fact_claim", "google_query": "test query", "newsapi_query": "test query"}',
            # verify_claim response - formatted verdict string
            """VERDICT: TRUE
CONFIDENCE: 0.9
REASONING: Good evidence found""",
            # summarize_input response
            "Executive summary with verification results included."
        ]

        mock_collect_evidence.return_value = [
            {
                'text': 'Test evidence',
                'url': 'http://example.com',
                'source': 'Test',
                'title': 'Test Article',
                'score': 0.9
            }
        ]

        result = check_text("Test claim to verify")

        # Verify structure
        assert 'summary' in result
        assert 'results' in result
        assert len(result['results']) == 1

        # Verify summary contains verification info
        assert "verification results" in result['summary'].lower() or "evidence" in result['summary'].lower()

    @patch('pipeline.summary.chat')
    @patch('search.base.collect_evidence')
    def test_multi_claim_pipeline_summary_timing(self, mock_collect_evidence, mock_chat):
        """Test that summary is generated after verification for multiple claims."""
        # Mock responses with proper string formats
        mock_chat.side_effect = [
            # detect_intent
            '{"intent": "multi_claim", "google_query": "multi claim query", "newsapi_query": "multi claim query"}',
            # extract_claims - return claims one per line with dashes
            """- AI will prevent job losses through automation
- AI will increase productivity in manufacturing""",
            # verify_claim 1
            """VERDICT: TRUE
CONFIDENCE: 0.9
REASONING: Evidence supports claim 1""",
            # verify_claim 2
            """VERDICT: MOSTLY_TRUE
CONFIDENCE: 0.7
REASONING: Evidence suggests net positive job impact""",
            # summarize_input
            "Comprehensive summary of multiple claims with verification results."
        ]

        mock_collect_evidence.return_value = [
            {
                'text': 'Test evidence',
                'url': 'http://example.com',
                'source': 'Test',
                'title': 'Test Article',
                'score': 0.9
            }
        ]

        result = check_text("Text with multiple claims to verify. This is a much longer text that contains several statements about artificial intelligence and its impact on society and the economy. The text discusses various aspects of AI development and deployment. Artificial intelligence will revolutionize healthcare by enabling better diagnostics and personalized treatment plans. Machine learning algorithms can predict disease outbreaks and help prevent pandemics. AI systems are being used in education to provide personalized learning experiences for students. Automation through AI will transform manufacturing processes and increase efficiency. Self-driving cars powered by AI will reduce traffic accidents and improve transportation safety. Natural language processing allows computers to understand and generate human-like text. Computer vision systems can analyze medical images with high accuracy. AI chatbots are improving customer service across many industries. Blockchain technology combined with AI creates new possibilities for secure data management. Quantum computing will exponentially increase AI processing capabilities. These technological advancements will create new job opportunities while also requiring workforce reskilling programs.")

        # Verify structure
        assert 'summary' in result
        assert 'results' in result
        assert len(result['results']) == 2

        # Verify summary was generated (timing test - content tested separately)
        assert len(result['summary']) > 0
        assert "comprehensive" in result['summary'].lower() or "summary" in result['summary'].lower()
@pytest.mark.live
class TestLiveAPIIntegration:
    """Live API tests for executive summary functionality."""

    def test_live_single_claim_summary(self):
        """Test executive summary with live API for single claim."""
        # This will use real APIs with system defaults
        result = check_text(
            "OpenAI released GPT-5 in January 2025",
            num_google=2,      # Limit API calls for testing
            num_news=2
        )

        assert 'summary' in result
        assert 'results' in result
        assert len(result['results']) > 0

        # Summary should exist (may be fallback if LLM fails)
        assert isinstance(result['summary'], str)
        assert len(result['summary']) > 0

        # If summary generation worked, it should include verification context
        if result['summary'] != "OpenAI released GPT-5 in January 2025":
            assert "verification" in result['summary'].lower() or "evidence" in result['summary'].lower()

    def test_live_multi_claim_summary(self):
        """Test executive summary with live API for multiple claims."""
        result = check_text(
            "Tesla achieved record profits in 2024 and Elon Musk announced a new Mars mission for 2026.",
            num_google=2,
            num_news=2
        )

        assert 'summary' in result
        assert 'results' in result
        assert len(result['results']) >= 1  # At least one claim should be found

        # Summary should mention multiple aspects
        summary_lower = result['summary'].lower()
        assert any(term in summary_lower for term in ['tesla', 'elon', 'mars', 'profits', 'mission'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
