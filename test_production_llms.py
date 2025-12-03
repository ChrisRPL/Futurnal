#!/usr/bin/env python
"""Test production LLMs (Qwen 2.5 32B Coder & Llama 3.3 70B) for extraction accuracy.

This script systematically tests both production models and compares results.
Run with: python test_production_llms.py
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_tests_with_model(model_name: str):
    """Run integration tests with specified production model."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING WITH: {model_name}")
    logger.info(f"{'='*80}\n")
    
    # Set environment variable
    env = os.environ.copy()
    env["FUTURNAL_PRODUCTION_LLM"] = model_name.lower()
    
    # Run pytest on key tests
    test_files = [
        "tests/extraction/integration/test_real_extraction_accuracy.py::test_event_extraction_accuracy_gate",
        "tests/extraction/integration/test_schema_alignment_gate.py::test_relationship_type_richness",
        "tests/extraction/integration/test_schema_alignment_gate.py::test_schema_evolution_semantic_alignment_gate",
    ]
    
    results = {}
    for test in test_files:
        test_name = test.split("::")[-1]
        logger.info(f"\n--- Running {test_name} ---")
        
        cmd = [
            "python", "-m", "pytest",
            test,
            "-xvs",
            "--tb=short"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout
            )
            
            passed = "PASSED" in result.stdout
            results[test_name] = "✅ PASSED" if passed else "❌ FAILED"
            
            # Show relevant output
            if not passed:
                # Extract error message
                for line in result.stdout.split('\n'):
                    if 'AssertionError' in line or 'accuracy' in line.lower():
                        logger.info(f"  {line.strip()}")
            else:
                logger.info(f"  ✅ Test passed!")
                
        except subprocess.TimeoutExpired:
            results[test_name] = "⏱️ TIMEOUT"
            logger.error(f"  Test timed out after 5 minutes")
        except Exception as e:
            results[test_name] = f"❌ ERROR: {str(e)}"
            logger.error(f"  Error running test: {e}")
    
    return results

def main():
    """Main test execution."""
    logger.info("Production LLM Comparison Test")
    logger.info("Testing: Qwen 2.5 32B Coder vs Llama 3.3 70B")
    logger.info(f"\nNOTE: This will download models if not cached.")
    logger.info("  - Qwen 2.5 32B: ~20GB (4-bit)")
    logger.info("  - Llama 3.3 70B: ~35GB (4-bit)")
    
    # Test both models
    qwen_results = run_tests_with_model("qwen")
    llama_results = run_tests_with_model("llama")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*80}\n")
    
    logger.info(f"{'Test':<50} {'Qwen 32B':<15} {'Llama 70B':<15}")
    logger.info(f"{'-'*50} {'-'*15} {'-'*15}")
    
    all_tests = set(list(qwen_results.keys()) + list(llama_results.keys()))
    for test in all_tests:
        qwen_res = qwen_results.get(test, "N/A")
        llama_res = llama_results.get(test, "N/A")
        logger.info(f"{test:<50} {qwen_res:<15} {llama_res:<15}")
    
    logger.info(f"\n{'='*80}")
    logger.info("RECOMMENDATION")
    logger.info(f"{'='*80}\n")
    
    qwen_passed = sum(1 for v in qwen_results.values() if "PASSED" in v)
    llama_passed = sum(1 for v in llama_results.values() if "PASSED" in v)
    
    if llama_passed > qwen_passed:
        logger.info("✅ Use Llama 3.3 70B for production (better accuracy)")
    elif qwen_passed > llama_passed:
        logger.info("✅ Use Qwen 2.5 32B Coder for production (good accuracy, less VRAM)")
    else:
        logger.info("⚖️ Both models perform similarly - use Qwen for less VRAM")
    
    logger.info(f"\nTo use selected model, set: export FUTURNAL_PRODUCTION_LLM=qwen (or llama)")
    logger.info(f"Or call: get_test_llm_client(fast=False, production_model='qwen')")

if __name__ == "__main__":
    main()
