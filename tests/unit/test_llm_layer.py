from __future__ import annotations
"""
test_llm_layer.py

Unit tests for the full LLM layer:
  - hallucination_check.py
  - prompt_builder.py      (explainer)
  - explainer.py           (with and without GEMINI_API_KEY)
  - output_parser.py       (judge)
  - prompt_builder.py      (judge)
  - judge.py               (with and without GROQ_API_KEY)

All tests run with zero API keys — the fallback paths are
a first-class feature of the design, not a workaround.
"""
import pytest
import json
import os
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from ingestion.schemas.transaction_event import TransactionEvent, RiskClass
from features.redis_client import FeatureStoreClient
from features.feature_pipeline import build_feature_vector
from prediction.ml_scorer.xgboost_scorer import score_transaction, ScorerOutput
from prediction.ml_scorer.shap_explainer import SHAPExplanation, FeatureContribution

from prediction.llm_explainer.hallucination_check import (
    verify_policy_citations, check_reason_codes,
)
from prediction.llm_explainer.prompt_builder import build_explanation_prompt, SYSTEM_PROMPT
from prediction.llm_explainer.explainer import generate_explanation, LLMExplainerOutput

from prediction.llm_judge.output_parser import parse_judge_response
from prediction.llm_judge.prompt_builder import build_judge_prompt, JUDGE_SYSTEM_PROMPT
from prediction.llm_judge.judge import run_judge, LLMJudgeOutput


# ── Shared fixtures ────────────────────────────────────────────────────────

def make_scorer_output(
    transaction_id: str = "TXN-LLM-001",
    risk_score: float = 0.92,
    risk_class: RiskClass = RiskClass.HIGH,
) -> ScorerOutput:
    shap = SHAPExplanation(
        top_features=[
            FeatureContribution(feature="destination_risk_score", value=5.0, contribution=5.63, abs_contribution=5.63),
            FeatureContribution(feature="is_amount_outlier",      value=1.0, contribution=1.33, abs_contribution=1.33),
            FeatureContribution(feature="risk_score_delta",       value=4.0, contribution=0.95, abs_contribution=0.95),
            FeatureContribution(feature="amount_zscore",          value=18.0,contribution=0.22, abs_contribution=0.22),
            FeatureContribution(feature="is_fatf_greylist",       value=1.0, contribution=0.18, abs_contribution=0.18),
        ],
        base_value=0.12,
        predicted_score=risk_score,
        model_version="xgboost_v1",
    )
    return ScorerOutput(
        transaction_id=transaction_id,
        risk_score=risk_score,
        risk_class=risk_class,
        shap_explanation=shap,
        model_version="xgboost_v1",
        scored_at=datetime.now(timezone.utc),
        latency_ms=1.87,
        scorer_mode="xgboost",
    )


SAMPLE_POLICIES = [
    {
        "policy_id": "AML-TM-04",
        "title": "Wire Transfer Velocity",
        "passage": "Wire transfers exceeding 3 standard deviations trigger mandatory review.",
        "version": "1.8",
        "jurisdiction": "ALL",
        "product": "WIRE_TRANSFER",
    },
    {
        "policy_id": "AML-KYC-02",
        "title": "PEP Enhanced Due Diligence",
        "passage": "PEP transactions over $25,000 require senior management approval.",
        "version": "3.1",
        "jurisdiction": "ALL",
        "product": "ALL",
    },
]

SAMPLE_TX_CONTEXT = {
    "transaction_id":           "TXN-LLM-001",
    "amount":                   95000.0,
    "currency":                 "USD",
    "channel":                  "WIRE",
    "jurisdiction_origin":      "US",
    "jurisdiction_destination": "AE",
    "timestamp":                "2026-06-13T02:30:00Z",
    "product_type":             "WIRE_TRANSFER",
}

SAMPLE_CUSTOMER_CONTEXT = {
    "customer_id":       "CUST-99",
    "risk_rating":       3,
    "prior_alert_count": 1,
    "account_age_days":  180,
}

SAMPLE_SANCTIONS = {
    "match_found":       False,
    "confidence":        0.0,
    "watchlist_source":  None,
}


# ══════════════════════════════════════════════════════════════════════════
# 1. hallucination_check.py
# ══════════════════════════════════════════════════════════════════════════

class TestHallucinationCheck:

    def test_valid_citation_passes_through(self):
        cited = [{"policy_id": "AML-TM-04", "passage": "some text"}]
        result = verify_policy_citations(cited, SAMPLE_POLICIES)
        assert len(result) == 1
        assert result[0]["policy_id"] == "AML-TM-04"

    def test_hallucinated_citation_is_dropped(self):
        cited = [{"policy_id": "AML-FAKE-99", "passage": "made up"}]
        result = verify_policy_citations(cited, SAMPLE_POLICIES)
        assert result == []

    def test_mix_of_real_and_fake_citations(self):
        cited = [
            {"policy_id": "AML-TM-04",   "passage": "real"},
            {"policy_id": "AML-FAKE-99", "passage": "fake"},
        ]
        result = verify_policy_citations(cited, SAMPLE_POLICIES)
        assert len(result) == 1
        assert result[0]["policy_id"] == "AML-TM-04"

    def test_empty_citations_returns_empty(self):
        assert verify_policy_citations([], SAMPLE_POLICIES) == []

    def test_empty_supplied_passages_drops_all(self):
        cited = [{"policy_id": "AML-TM-04", "passage": "text"}]
        result = verify_policy_citations(cited, [])
        assert result == []

    def test_multiple_valid_citations_all_pass(self):
        cited = [
            {"policy_id": "AML-TM-04",  "passage": "p1"},
            {"policy_id": "AML-KYC-02", "passage": "p2"},
        ]
        result = verify_policy_citations(cited, SAMPLE_POLICIES)
        assert len(result) == 2

    def test_valid_reason_codes_pass(self):
        codes = ["AML_VELOCITY", "AMOUNT_DEVIATION", "HIGH_RISK_JURISDICTION"]
        result = check_reason_codes(codes)
        assert result == codes

    def test_unknown_reason_code_is_dropped(self):
        codes = ["AML_VELOCITY", "MADE_UP_CODE", "AMOUNT_DEVIATION"]
        result = check_reason_codes(codes)
        assert "MADE_UP_CODE" not in result
        assert "AML_VELOCITY" in result
        assert "AMOUNT_DEVIATION" in result

    def test_empty_reason_codes_returns_empty(self):
        assert check_reason_codes([]) == []

    def test_all_valid_reason_codes_accepted(self):
        all_codes = [
            "AML_VELOCITY", "AMOUNT_DEVIATION", "HIGH_RISK_JURISDICTION",
            "SANCTIONS_PROXIMITY", "PEP_RISK", "STRUCTURING_PATTERN",
            "NEW_BENEFICIARY", "CHANNEL_ANOMALY", "ROUND_AMOUNT",
            "PRIOR_ALERTS", "HIGH_RISK_CUSTOMER", "FATF_BLACKLIST", "FATF_GREYLIST",
        ]
        result = check_reason_codes(all_codes)
        assert set(result) == set(all_codes)

    def test_custom_allowed_codes(self):
        codes = ["CUSTOM_CODE", "AML_VELOCITY"]
        result = check_reason_codes(codes, allowed_codes={"CUSTOM_CODE"})
        assert result == ["CUSTOM_CODE"]


# ══════════════════════════════════════════════════════════════════════════
# 2. explainer prompt_builder.py
# ══════════════════════════════════════════════════════════════════════════

class TestExplainerPromptBuilder:

    def test_prompt_contains_transaction_id(self):
        scored = make_scorer_output()
        prompt = build_explanation_prompt(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert "TXN-LLM-001" in prompt or "95000" in prompt

    def test_prompt_contains_risk_score(self):
        scored = make_scorer_output(risk_score=0.92)
        prompt = build_explanation_prompt(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert "0.92" in prompt or "HIGH" in prompt

    def test_prompt_contains_policy_ids(self):
        scored = make_scorer_output()
        prompt = build_explanation_prompt(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert "AML-TM-04" in prompt

    def test_prompt_contains_top_features(self):
        scored = make_scorer_output()
        prompt = build_explanation_prompt(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert "destination_risk_score" in prompt

    def test_prompt_contains_jurisdiction(self):
        scored = make_scorer_output()
        prompt = build_explanation_prompt(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert "AE" in prompt or "USD" in prompt

    def test_empty_policies_handled_gracefully(self):
        scored = make_scorer_output()
        prompt = build_explanation_prompt(scored, [], SAMPLE_TX_CONTEXT)
        assert "No matching policy passages found" in prompt

    def test_system_prompt_loaded(self):
        assert len(SYSTEM_PROMPT) > 100
        assert "JSON" in SYSTEM_PROMPT
        assert "policy" in SYSTEM_PROMPT.lower()

    def test_system_prompt_forbids_decision(self):
        assert "recommend a decision" in SYSTEM_PROMPT.lower() or \
               "reviewer" in SYSTEM_PROMPT.lower()

    def test_prompt_is_string(self):
        scored = make_scorer_output()
        prompt = build_explanation_prompt(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_max_3_policy_passages_in_prompt(self):
        many_policies = SAMPLE_POLICIES * 5   # 10 passages
        scored = make_scorer_output()
        prompt = build_explanation_prompt(scored, many_policies, SAMPLE_TX_CONTEXT)
        # Only first 3 should appear — count distinct policy_id occurrences
        assert prompt.count("AML-TM-04") <= 3


# ══════════════════════════════════════════════════════════════════════════
# 3. explainer.py — fallback path (no API key)
# ══════════════════════════════════════════════════════════════════════════

class TestLLMExplainerFallback:
    """Tests the fallback path when GEMINI_API_KEY is not set."""

    def test_returns_explainer_output(self):
        scored = make_scorer_output()
        result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert isinstance(result, LLMExplainerOutput)

    def test_fallback_used_true_without_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            scored = make_scorer_output()
            result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
            assert result.fallback_used is True

    def test_transaction_id_preserved(self):
        scored = make_scorer_output(transaction_id="TXN-ID-TEST")
        result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert result.transaction_id == "TXN-ID-TEST"

    def test_explanation_text_is_non_empty(self):
        scored = make_scorer_output()
        result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert len(result.explanation_text) > 10

    def test_reason_codes_are_valid(self):
        scored = make_scorer_output()
        result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        valid = {
            "AML_VELOCITY", "AMOUNT_DEVIATION", "HIGH_RISK_JURISDICTION",
            "SANCTIONS_PROXIMITY", "PEP_RISK", "STRUCTURING_PATTERN",
            "NEW_BENEFICIARY", "CHANNEL_ANOMALY", "ROUND_AMOUNT",
            "PRIOR_ALERTS", "HIGH_RISK_CUSTOMER", "FATF_BLACKLIST", "FATF_GREYLIST",
        }
        for code in result.reason_codes:
            assert code in valid, f"Invalid reason code: {code}"

    def test_prompt_hash_is_set(self):
        scored = make_scorer_output()
        result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert len(result.prompt_hash) == 64     # SHA-256 hex = 64 chars

    def test_prompt_hash_deterministic(self):
        """Same input → same hash every time."""
        scored = make_scorer_output()
        r1 = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        r2 = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert r1.prompt_hash == r2.prompt_hash

    def test_generated_at_is_recent(self):
        scored = make_scorer_output()
        result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        now = datetime.now(timezone.utc)
        delta = abs((now - result.generated_at).total_seconds())
        assert delta < 5

    def test_model_id_set_in_fallback(self):
        scored = make_scorer_output()
        result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert result.model_id != ""

    def test_output_serialises_to_json(self):
        scored = make_scorer_output()
        result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        json_str = result.model_dump_json()
        assert "explanation_text" in json_str
        assert "reason_codes" in json_str
        assert "prompt_hash" in json_str


# ══════════════════════════════════════════════════════════════════════════
# 4. explainer.py — mocked Gemini API call
# ══════════════════════════════════════════════════════════════════════════

class TestLLMExplainerWithMockedAPI:
    """Tests the live path by mocking the Gemini client."""

    VALID_RESPONSE = json.dumps({
        "explanation_text": "This $95,000 wire to AE is 18 standard deviations above the customer's average, triggering AML-TM-04 mandatory review threshold.",
        "reason_codes": ["AMOUNT_DEVIATION", "HIGH_RISK_JURISDICTION"],
        "policy_citations": [
            {"policy_id": "AML-TM-04", "passage": "Wire transfers exceeding 3 standard deviations trigger mandatory review."}
        ],
    })

    def _make_mock_client(self, response_text: str):
        mock_response = MagicMock()
        mock_response.text = response_text
        mock_client = MagicMock()
        mock_client.generate_content.return_value = mock_response
        return mock_client

    def test_uses_llm_response_when_api_key_set(self):
        mock_client = self._make_mock_client(self.VALID_RESPONSE)
        scored = make_scorer_output()
        with patch("prediction.llm_explainer.explainer._get_gemini_client",
                   return_value=mock_client):
            result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert result.fallback_used is False
        assert result.model_id == "gemini-2.5-flash"

    def test_explanation_text_from_llm(self):
        mock_client = self._make_mock_client(self.VALID_RESPONSE)
        scored = make_scorer_output()
        with patch("prediction.llm_explainer.explainer._get_gemini_client",
                   return_value=mock_client):
            result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert "AML-TM-04" in result.explanation_text or \
               "standard deviations" in result.explanation_text

    def test_hallucinated_policy_stripped_from_live_response(self):
        response_with_fake = json.dumps({
            "explanation_text": "High risk transaction.",
            "reason_codes": ["AMOUNT_DEVIATION"],
            "policy_citations": [
                {"policy_id": "AML-FAKE-99", "passage": "hallucinated policy"},
                {"policy_id": "AML-TM-04",   "passage": "real policy"},
            ],
        })
        mock_client = self._make_mock_client(response_with_fake)
        scored = make_scorer_output()
        with patch("prediction.llm_explainer.explainer._get_gemini_client",
                   return_value=mock_client):
            result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        policy_ids = [c["policy_id"] for c in result.policy_citations]
        assert "AML-FAKE-99" not in policy_ids
        assert "AML-TM-04" in policy_ids

    def test_invalid_json_falls_back_gracefully(self):
        mock_client = self._make_mock_client("this is not JSON at all")
        scored = make_scorer_output()
        with patch("prediction.llm_explainer.explainer._get_gemini_client",
                   return_value=mock_client):
            result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert result.fallback_used is True

    def test_api_exception_falls_back_gracefully(self):
        mock_client = MagicMock()
        mock_client.generate_content.side_effect = Exception("API timeout")
        scored = make_scorer_output()
        with patch("prediction.llm_explainer.explainer._get_gemini_client",
                   return_value=mock_client):
            result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert result.fallback_used is True
        assert result.transaction_id == scored.transaction_id

    def test_markdown_fences_stripped_from_response(self):
        fenced = "```json\n" + self.VALID_RESPONSE + "\n```"
        mock_client = self._make_mock_client(fenced)
        scored = make_scorer_output()
        with patch("prediction.llm_explainer.explainer._get_gemini_client",
                   return_value=mock_client):
            result = generate_explanation(scored, SAMPLE_POLICIES, SAMPLE_TX_CONTEXT)
        assert result.fallback_used is False


# ══════════════════════════════════════════════════════════════════════════
# 5. judge output_parser.py
# ══════════════════════════════════════════════════════════════════════════

class TestJudgeOutputParser:

    def test_valid_json_parsed_correctly(self):
        raw = json.dumps({
            "recommendation": "ESCALATE",
            "confidence": "HIGH",
            "narrative": "Large wire to sanctioned-adjacent jurisdiction. Warrants immediate review.",
            "supporting_signals": ["sig one", "sig two", "sig three"],
        })
        result = parse_judge_response(raw)
        assert result["recommendation"] == "ESCALATE"
        assert result["confidence"] == "HIGH"
        assert len(result["supporting_signals"]) == 3

    def test_all_valid_recommendations_accepted(self):
        for rec in ("ESCALATE", "FLAG", "MONITOR", "CLOSE"):
            raw = json.dumps({
                "recommendation": rec,
                "confidence": "HIGH",
                "narrative": "test",
                "supporting_signals": ["a", "b", "c"],
            })
            result = parse_judge_response(raw)
            assert result["recommendation"] == rec

    def test_all_valid_confidences_accepted(self):
        for conf in ("HIGH", "MEDIUM", "LOW"):
            raw = json.dumps({
                "recommendation": "FLAG",
                "confidence": conf,
                "narrative": "test",
                "supporting_signals": ["a", "b", "c"],
            })
            result = parse_judge_response(raw)
            assert result["confidence"] == conf

    def test_invalid_recommendation_defaults_to_flag(self):
        raw = json.dumps({
            "recommendation": "BANANA",
            "confidence": "HIGH",
            "narrative": "test",
            "supporting_signals": ["a", "b", "c"],
        })
        result = parse_judge_response(raw)
        assert result["recommendation"] == "FLAG"

    def test_invalid_confidence_defaults_to_low(self):
        raw = json.dumps({
            "recommendation": "FLAG",
            "confidence": "VERY_HIGH",
            "narrative": "test",
            "supporting_signals": ["a", "b", "c"],
        })
        result = parse_judge_response(raw)
        assert result["confidence"] == "LOW"

    def test_broken_json_returns_safe_default(self):
        result = parse_judge_response("this is not json {{{")
        assert result["recommendation"] == "FLAG"
        assert result["confidence"] == "LOW"
        assert len(result["supporting_signals"]) == 3

    def test_empty_string_returns_safe_default(self):
        result = parse_judge_response("")
        assert result["recommendation"] == "FLAG"

    def test_signals_padded_to_3_if_fewer(self):
        raw = json.dumps({
            "recommendation": "FLAG",
            "confidence": "LOW",
            "narrative": "test",
            "supporting_signals": ["only one"],
        })
        result = parse_judge_response(raw)
        assert len(result["supporting_signals"]) == 3

    def test_signals_capped_at_3_if_more(self):
        raw = json.dumps({
            "recommendation": "FLAG",
            "confidence": "LOW",
            "narrative": "test",
            "supporting_signals": ["a", "b", "c", "d", "e"],
        })
        result = parse_judge_response(raw)
        assert len(result["supporting_signals"]) == 3

    def test_narrative_capped_at_500_chars(self):
        long_narrative = "x" * 1000
        raw = json.dumps({
            "recommendation": "FLAG",
            "confidence": "LOW",
            "narrative": long_narrative,
            "supporting_signals": ["a", "b", "c"],
        })
        result = parse_judge_response(raw)
        assert len(result["narrative"]) <= 500

    def test_markdown_fences_stripped(self):
        inner = json.dumps({
            "recommendation": "ESCALATE",
            "confidence": "HIGH",
            "narrative": "test",
            "supporting_signals": ["a", "b", "c"],
        })
        fenced = f"```json\n{inner}\n```"
        result = parse_judge_response(fenced)
        assert result["recommendation"] == "ESCALATE"

    def test_lowercase_recommendation_uppercased(self):
        raw = json.dumps({
            "recommendation": "escalate",
            "confidence": "high",
            "narrative": "test",
            "supporting_signals": ["a", "b", "c"],
        })
        result = parse_judge_response(raw)
        assert result["recommendation"] == "ESCALATE"
        assert result["confidence"] == "HIGH"


# ══════════════════════════════════════════════════════════════════════════
# 6. judge prompt_builder.py
# ══════════════════════════════════════════════════════════════════════════

class TestJudgePromptBuilder:

    def _make_explainer_output(self) -> LLMExplainerOutput:
        return LLMExplainerOutput(
            transaction_id="TXN-LLM-001",
            explanation_text="High-risk wire to AE exceeding customer average by 18x.",
            reason_codes=["AMOUNT_DEVIATION", "HIGH_RISK_JURISDICTION"],
            policy_citations=[{"policy_id": "AML-TM-04", "passage": "..."}],
            model_id="fallback",
            prompt_hash="abc123",
            generated_at=datetime.now(timezone.utc),
            fallback_used=True,
        )

    def test_prompt_contains_risk_score(self):
        scored = make_scorer_output(risk_score=0.92)
        expl   = self._make_explainer_output()
        prompt = build_judge_prompt(
            scored, expl, SAMPLE_CUSTOMER_CONTEXT,
            SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert "0.92" in prompt or "0.9" in prompt

    def test_prompt_contains_risk_class(self):
        scored = make_scorer_output()
        expl   = self._make_explainer_output()
        prompt = build_judge_prompt(
            scored, expl, SAMPLE_CUSTOMER_CONTEXT,
            SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert "HIGH" in prompt

    def test_prompt_contains_explainer_text(self):
        scored = make_scorer_output()
        expl   = self._make_explainer_output()
        prompt = build_judge_prompt(
            scored, expl, SAMPLE_CUSTOMER_CONTEXT,
            SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert "18x" in prompt or "AE" in prompt

    def test_prompt_contains_customer_risk_rating(self):
        scored = make_scorer_output()
        expl   = self._make_explainer_output()
        prompt = build_judge_prompt(
            scored, expl, SAMPLE_CUSTOMER_CONTEXT,
            SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert "3" in prompt   # risk_rating=3

    def test_sanctions_match_shown_when_found(self):
        scored  = make_scorer_output()
        expl    = self._make_explainer_output()
        hit     = {"match_found": True, "confidence": 0.92, "watchlist_source": "OFAC-SDN"}
        prompt  = build_judge_prompt(
            scored, expl, SAMPLE_CUSTOMER_CONTEXT,
            hit, [], SAMPLE_TX_CONTEXT,
        )
        assert "MATCH FOUND" in prompt or "0.92" in prompt

    def test_no_sanctions_match_shown_when_clean(self):
        scored = make_scorer_output()
        expl   = self._make_explainer_output()
        prompt = build_judge_prompt(
            scored, expl, SAMPLE_CUSTOMER_CONTEXT,
            SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert "No sanctions match" in prompt

    def test_similar_cases_included_when_present(self):
        scored = make_scorer_output()
        expl   = self._make_explainer_output()
        cases  = [
            {"case_id": "CASE-001", "risk_class": "HIGH",
             "disposition": "ESCALATED", "key_signal": "FATF jurisdiction"},
        ]
        prompt = build_judge_prompt(
            scored, expl, SAMPLE_CUSTOMER_CONTEXT,
            SAMPLE_SANCTIONS, cases, SAMPLE_TX_CONTEXT,
        )
        assert "CASE-001" in prompt or "ESCALATED" in prompt

    def test_no_cases_message_when_empty(self):
        scored = make_scorer_output()
        expl   = self._make_explainer_output()
        prompt = build_judge_prompt(
            scored, expl, SAMPLE_CUSTOMER_CONTEXT,
            SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert "No similar historical cases" in prompt

    def test_judge_system_prompt_loaded(self):
        assert len(JUDGE_SYSTEM_PROMPT) > 100
        assert "ESCALATE" in JUDGE_SYSTEM_PROMPT
        assert "JSON" in JUDGE_SYSTEM_PROMPT

    def test_judge_prompt_is_string(self):
        scored = make_scorer_output()
        expl   = self._make_explainer_output()
        prompt = build_judge_prompt(
            scored, expl, SAMPLE_CUSTOMER_CONTEXT,
            SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 200


# ══════════════════════════════════════════════════════════════════════════
# 7. judge.py — fallback path (no API key)
# ══════════════════════════════════════════════════════════════════════════

class TestLLMJudgeFallback:

    def _explainer_output(self) -> LLMExplainerOutput:
        return LLMExplainerOutput(
            transaction_id="TXN-LLM-001",
            explanation_text="Test explanation.",
            reason_codes=["AMOUNT_DEVIATION"],
            policy_citations=[],
            model_id="fallback",
            prompt_hash="abc123",
            generated_at=datetime.now(timezone.utc),
            fallback_used=True,
        )

    def test_returns_judge_output(self):
        scored = make_scorer_output()
        result = run_judge(
            scored, self._explainer_output(),
            SAMPLE_CUSTOMER_CONTEXT, SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert isinstance(result, LLMJudgeOutput)

    def test_fallback_used_without_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            scored = make_scorer_output()
            result = run_judge(
                scored, self._explainer_output(),
                SAMPLE_CUSTOMER_CONTEXT, SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
            )
            assert result.fallback_used is True

    def test_high_risk_fallback_recommends_escalate(self):
        scored = make_scorer_output(risk_class=RiskClass.HIGH)
        result = run_judge(
            scored, self._explainer_output(),
            SAMPLE_CUSTOMER_CONTEXT, SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert result.recommendation == "ESCALATE"

    def test_medium_risk_fallback_recommends_flag(self):
        scored = make_scorer_output(risk_score=0.55, risk_class=RiskClass.MEDIUM)
        result = run_judge(
            scored, self._explainer_output(),
            SAMPLE_CUSTOMER_CONTEXT, SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert result.recommendation == "FLAG"

    def test_low_risk_fallback_recommends_monitor(self):
        scored = make_scorer_output(risk_score=0.20, risk_class=RiskClass.LOW)
        result = run_judge(
            scored, self._explainer_output(),
            SAMPLE_CUSTOMER_CONTEXT, SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert result.recommendation == "MONITOR"

    def test_transaction_id_preserved(self):
        scored = make_scorer_output(transaction_id="TXN-JUDGE-ID")
        result = run_judge(
            scored, self._explainer_output(),
            SAMPLE_CUSTOMER_CONTEXT, SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert result.transaction_id == "TXN-JUDGE-ID"

    def test_prompt_hash_is_64_chars(self):
        scored = make_scorer_output()
        result = run_judge(
            scored, self._explainer_output(),
            SAMPLE_CUSTOMER_CONTEXT, SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert len(result.prompt_hash) == 64

    def test_exactly_3_supporting_signals(self):
        scored = make_scorer_output()
        result = run_judge(
            scored, self._explainer_output(),
            SAMPLE_CUSTOMER_CONTEXT, SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        assert len(result.supporting_signals) == 3

    def test_output_serialises_to_json(self):
        scored = make_scorer_output()
        result = run_judge(
            scored, self._explainer_output(),
            SAMPLE_CUSTOMER_CONTEXT, SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
        )
        json_str = result.model_dump_json()
        assert "recommendation" in json_str
        assert "narrative" in json_str
        assert "supporting_signals" in json_str


# ══════════════════════════════════════════════════════════════════════════
# 8. judge.py — mocked Groq API call
# ══════════════════════════════════════════════════════════════════════════

class TestLLMJudgeWithMockedAPI:

    VALID_RESPONSE = json.dumps({
        "recommendation": "ESCALATE",
        "confidence": "HIGH",
        "narrative": "Wire of $95K to UAE is 18x customer average with FATF greylist exposure. Prior alert history warrants immediate escalation.",
        "supporting_signals": [
            "Amount is 18x customer 90-day average (z-score: 18)",
            "Destination UAE is on FATF greylist",
            "Customer has 1 prior compliance alert",
        ],
    })

    def _make_mock_groq(self, content: str):
        mock_choice  = MagicMock()
        mock_choice.message.content = content
        mock_resp    = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client  = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        return mock_client

    def _expl(self) -> LLMExplainerOutput:
        return LLMExplainerOutput(
            transaction_id="TXN-LLM-001",
            explanation_text="High risk.",
            reason_codes=["AMOUNT_DEVIATION"],
            policy_citations=[],
            model_id="gemini-2.5-flash",
            prompt_hash="abc123",
            generated_at=datetime.now(timezone.utc),
            fallback_used=False,
        )

    def test_live_path_not_fallback(self):
        mock_groq = self._make_mock_groq(self.VALID_RESPONSE)
        scored    = make_scorer_output()
        with patch("prediction.llm_judge.judge._get_groq_client",
                   return_value=mock_groq):
            result = run_judge(
                scored, self._expl(),
                SAMPLE_CUSTOMER_CONTEXT, SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
            )
        assert result.fallback_used is False
        assert result.model_id == "llama-3.3-70b-versatile"

    def test_recommendation_from_groq_response(self):
        mock_groq = self._make_mock_groq(self.VALID_RESPONSE)
        scored    = make_scorer_output()
        with patch("prediction.llm_judge.judge._get_groq_client",
                   return_value=mock_groq):
            result = run_judge(
                scored, self._expl(),
                SAMPLE_CUSTOMER_CONTEXT, SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
            )
        assert result.recommendation == "ESCALATE"
        assert result.confidence == "HIGH"

    def test_groq_api_error_falls_back(self):
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = Exception("connection refused")
        scored = make_scorer_output()
        with patch("prediction.llm_judge.judge._get_groq_client",
                   return_value=mock_groq):
            result = run_judge(
                scored, self._expl(),
                SAMPLE_CUSTOMER_CONTEXT, SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
            )
        assert result.fallback_used is True

    def test_invalid_json_from_groq_falls_back(self):
        mock_groq = self._make_mock_groq("not valid json {{")
        scored    = make_scorer_output()
        with patch("prediction.llm_judge.judge._get_groq_client",
                   return_value=mock_groq):
            result = run_judge(
                scored, self._expl(),
                SAMPLE_CUSTOMER_CONTEXT, SAMPLE_SANCTIONS, [], SAMPLE_TX_CONTEXT,
            )
        assert result.recommendation in ("ESCALATE", "FLAG", "MONITOR", "CLOSE")
        assert len(result.supporting_signals) == 3


# ══════════════════════════════════════════════════════════════════════════
# 9. End-to-end integration — full pipeline through LLM layer
# ══════════════════════════════════════════════════════════════════════════

class TestLLMLayerIntegration:

    def test_full_pipeline_high_risk_no_api_keys(self):
        """Complete feature → score → explain → judge pipeline, no API keys."""
        event = TransactionEvent(
            transaction_id="TXN-E2E-HIGH",
            customer_id="CUST-E2E-01",
            amount=150000.0, currency="USD", channel="WIRE",
            origin_account="ACC-US-1", destination_account="ACC-RU-9",
            jurisdiction_origin="US", jurisdiction_destination="RU",
            product_type="WIRE_TRANSFER",
            timestamp=datetime(2026, 6, 13, 2, 0, 0, tzinfo=timezone.utc),
        )
        store = FeatureStoreClient()
        store.seed_customer("CUST-E2E-01",
            avg_amount=5000.0, risk_rating=4,
            prior_alerts=2, account_age_days=90,
        )
        fv     = build_feature_vector(event, store)
        scored = score_transaction(fv, "WIRE_TRANSFER", "RU")

        tx_ctx   = {"amount": 150000, "currency": "USD", "channel": "WIRE",
                    "jurisdiction_origin": "US", "jurisdiction_destination": "RU",
                    "timestamp": "2026-06-13T02:00:00Z"}
        policies = [{"policy_id": "AML-TM-04", "version": "1.8",
                     "passage": "Outbound wire transfers over 3 std devs trigger review."}]
        cust     = {"risk_rating": 4, "prior_alert_count": 2, "account_age_days": 90}
        sanctions = {"match_found": False, "confidence": 0.0}

        expl  = generate_explanation(scored, policies, tx_ctx)
        judge = run_judge(scored, expl, cust, sanctions, [], tx_ctx)

        # Assertions on the full chain
        assert scored.risk_class == RiskClass.HIGH
        assert expl.transaction_id == "TXN-E2E-HIGH"
        assert expl.prompt_hash != ""
        assert judge.transaction_id == "TXN-E2E-HIGH"
        assert judge.recommendation in ("ESCALATE", "FLAG", "MONITOR", "CLOSE")
        assert len(judge.supporting_signals) == 3

    def test_low_risk_skips_llm_gracefully(self):
        """Low-risk transactions don't need LLM — but calling it still works."""
        event = TransactionEvent(
            transaction_id="TXN-E2E-LOW",
            customer_id="CUST-E2E-02",
            amount=500.0, currency="USD", channel="ACH",
            origin_account="ACC-US-1", destination_account="ACC-US-2",
            jurisdiction_origin="US", jurisdiction_destination="US",
            product_type="ACH_PAYMENT",
            timestamp=datetime(2026, 6, 13, 11, 0, 0, tzinfo=timezone.utc),
        )
        store = FeatureStoreClient()
        store.seed_customer("CUST-E2E-02",
            avg_amount=600.0, risk_rating=1,
            prior_alerts=0, account_age_days=730,
        )
        fv     = build_feature_vector(event, store)
        scored = score_transaction(fv, "ACH_PAYMENT", "US")

        assert scored.risk_class == RiskClass.LOW
        # Explainer still works for LOW — just returns sensible fallback
        expl = generate_explanation(scored, [], {
            "amount": 500, "currency": "USD", "channel": "ACH",
            "jurisdiction_origin": "US", "jurisdiction_destination": "US",
            "timestamp": "2026-06-13T11:00:00Z",
        })
        assert expl.transaction_id == "TXN-E2E-LOW"
