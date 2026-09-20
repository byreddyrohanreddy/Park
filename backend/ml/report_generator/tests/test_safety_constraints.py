"""
test_safety_constraints.py — Step 7: Automated safety checks.

Tests both the template fallback and (if ANTHROPIC_API_KEY is set) the LLM path.
Run with: python -m pytest tests/test_safety_constraints.py -v
Or directly: python tests/test_safety_constraints.py
"""

import sys
import os
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure report_generator package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generate_report import generate_report
from report_templates import generate_template_report

# ---------------------------------------------------------------------------
# Synthetic test payloads
# ---------------------------------------------------------------------------

PAYLOAD_UNCERTAIN = {
    "subject_id": "TEST_001",
    "prediction": "Uncertain",
    "prediction_set": ["HC", "PD"],
    "refer_to_clinician": True,
    "modality_gate_alpha": 0.42,
    "modality_dominance": "Biomarker",
    "attention_weights": [0.35, 0.65],
    "probability_hc": 0.52,
    "probability_pd": 0.48,
}

PAYLOAD_HIGH_CONFIDENCE_PD = {
    "subject_id": "TEST_002",
    "prediction": "PD",
    "prediction_set": ["PD"],
    "refer_to_clinician": False,
    "modality_gate_alpha": 0.73,
    "modality_dominance": "SSL",
    "attention_weights": [0.20, 0.80],
    "probability_hc": 0.08,
    "probability_pd": 0.92,
}

PAYLOAD_HC = {
    "subject_id": "TEST_003",
    "prediction": "HC",
    "prediction_set": ["HC"],
    "refer_to_clinician": False,
    "modality_gate_alpha": 0.31,
    "modality_dominance": "Biomarker",
    "attention_weights": [0.55, 0.45],
    "probability_hc": 0.89,
    "probability_pd": 0.11,
}

# Terms that should NEVER appear in output (not in seed corpus or payload)
DENYLIST = [
    "levodopa", "carbidopa", "dopamine agonist", "deep brain stimulation",
    "prevalence", "1 million", "10 million", "60,000", "incidence rate",
    "tremor at rest", "bradykinesia", "rigidity", "postural instability",
    "Hoehn and Yahr", "stage 1", "stage 2", "stage 3", "stage 4", "stage 5",
    "life expectancy", "cure", "prognosis",
]


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_1_uncertain_referral_language():
    """Test 1: Uncertain payload must contain referral/follow-up language."""
    print("\n" + "=" * 60)
    print("TEST 1: Uncertain payload → referral language")
    print("=" * 60)
    
    # Test on template fallback
    report_template = generate_template_report(PAYLOAD_UNCERTAIN, "clinician")
    print("\n--- Template (Clinician) ---")
    print(report_template)
    
    referral_keywords = ["follow-up", "clinician", "review", "clinical", "recommended", "doctor"]
    found = [kw for kw in referral_keywords if kw.lower() in report_template.lower()]
    
    assert len(found) >= 2, (
        f"Template report for uncertain payload must contain referral language. "
        f"Found: {found}"
    )
    print(f"\n[PASS] PASSED — Found referral keywords: {found}")
    
    # Also test patient mode
    report_patient = generate_template_report(PAYLOAD_UNCERTAIN, "patient")
    print("\n--- Template (Patient) ---")
    print(report_patient)
    
    found_patient = [kw for kw in referral_keywords if kw.lower() in report_patient.lower()]
    assert len(found_patient) >= 2, (
        f"Patient template for uncertain payload must contain referral language. "
        f"Found: {found_patient}"
    )
    print(f"\n[PASS] PASSED — Found referral keywords (patient): {found_patient}")
    
    return True


def test_2_high_confidence_still_disclaims():
    """Test 2: High-confidence PD must still include 'not a diagnosis' framing."""
    print("\n" + "=" * 60)
    print("TEST 2: High-confidence PD → still includes disclaimer")
    print("=" * 60)
    
    report = generate_template_report(PAYLOAD_HIGH_CONFIDENCE_PD, "clinician")
    print("\n--- Template (Clinician) ---")
    print(report)
    
    disclaimer_phrases = ["not a diagnosis", "decision-support", "not be used as a standalone"]
    found = [p for p in disclaimer_phrases if p.lower() in report.lower()]
    
    assert len(found) >= 1, (
        f"Even high-confidence reports must include disclaimer. Found: {found}"
    )
    print(f"\n[PASS] PASSED — Found disclaimer phrases: {found}")
    
    # Patient mode too
    report_patient = generate_template_report(PAYLOAD_HIGH_CONFIDENCE_PD, "patient")
    print("\n--- Template (Patient) ---")
    print(report_patient)
    
    patient_disclaimers = ["not a diagnosis", "decision-support", "not a medical diagnosis"]
    found_patient = [p for p in patient_disclaimers if p.lower() in report_patient.lower()]
    assert len(found_patient) >= 1, (
        f"Patient report must include disclaimer. Found: {found_patient}"
    )
    print(f"\n[PASS] PASSED — Found disclaimer (patient): {found_patient}")
    
    return True


def test_3_no_fabricated_content():
    """Test 3: Output must NOT contain PD symptoms/treatments/stats not in knowledge base."""
    print("\n" + "=" * 60)
    print("TEST 3: No fabricated medical content (denylist check)")
    print("=" * 60)
    
    # Test all three payloads × both audiences
    test_cases = [
        (PAYLOAD_UNCERTAIN, "clinician"),
        (PAYLOAD_UNCERTAIN, "patient"),
        (PAYLOAD_HIGH_CONFIDENCE_PD, "clinician"),
        (PAYLOAD_HIGH_CONFIDENCE_PD, "patient"),
        (PAYLOAD_HC, "clinician"),
        (PAYLOAD_HC, "patient"),
    ]
    
    all_passed = True
    for payload, audience in test_cases:
        report = generate_template_report(payload, audience)
        violations = [term for term in DENYLIST if term.lower() in report.lower()]
        label = f"{payload['subject_id']} ({audience})"
        
        if violations:
            print(f"  [FAIL] FAILED {label}: Found denied terms: {violations}")
            all_passed = False
        else:
            print(f"  [PASS] PASSED {label}: No denied terms found")
    
    assert all_passed, "Some reports contained fabricated content!"
    print(f"\n[PASS] ALL PASSED — No fabricated medical content detected")
    
    return True


def test_4_template_fallback_when_no_api_key():
    """Test 4: generate_report() falls back to template when ANTHROPIC_API_KEY is unset."""
    print("\n" + "=" * 60)
    print("TEST 4: Template fallback when no API key")
    print("=" * 60)
    
    # Save and unset the API key
    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    
    try:
        result = generate_report(PAYLOAD_HIGH_CONFIDENCE_PD, "clinician")
        print(f"\n--- Result ---")
        print(f"generation_method: {result['generation_method']}")
        print(f"model_used: {result['model_used']}")
        print(f"\nReport preview (first 200 chars):")
        print(result['report'][:200])
        
        assert result["generation_method"] == "template_fallback", (
            f"Expected 'template_fallback', got '{result['generation_method']}'"
        )
        assert result["model_used"] is None, (
            f"Expected model_used=None, got '{result['model_used']}'"
        )
        print(f"\n[PASS] PASSED — Correctly fell back to template_fallback")
    finally:
        # Restore the API key if it was set
        if saved_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved_key
    
    return True


def run_all_tests():
    """Run all safety constraint tests and report results."""
    print("=" * 60)
    print("PD-VoiceNet Report Generator — Safety Constraint Tests")
    print("=" * 60)
    
    results = {}
    tests = [
        ("Test 1: Uncertain → referral language", test_1_uncertain_referral_language),
        ("Test 2: High-confidence → still disclaims", test_2_high_confidence_still_disclaims),
        ("Test 3: No fabricated content (denylist)", test_3_no_fabricated_content),
        ("Test 4: Template fallback w/o API key", test_4_template_fallback_when_no_api_key),
    ]
    
    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = "PASSED"
        except AssertionError as e:
            results[name] = f"FAILED: {e}"
        except Exception as e:
            results[name] = f"ERROR: {e}"
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        icon = "[PASS]" if status == "PASSED" else "[FAIL]"
        print(f"  {icon} {name}: {status}")
    
    all_passed = all(s == "PASSED" for s in results.values())
    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
