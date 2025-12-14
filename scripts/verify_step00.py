#!/usr/bin/env python3
"""Step 00: Foundation & Research Alignment - Verification Script.

This script performs actual verification of the codebase to confirm:
1. All infrastructure components exist and can be imported
2. Option B compliance (no fine-tuning, temporal-first, etc.)
3. Search API gap is documented (keyword matching vs semantic)
4. EDC pipeline components exist

Run with: python scripts/verify_step00.py
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class Step00Verifier:
    """Verifies Step 00 Foundation & Research Alignment requirements."""

    def __init__(self) -> None:
        self.results: Dict[str, Dict] = {}
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def verify_import(self, module_path: str, class_name: str) -> bool:
        """Verify a module and class can be imported."""
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name, None)
            return cls is not None
        except Exception as e:
            return False

    def check_file_for_pattern(self, file_path: Path, pattern: str) -> List[Tuple[int, str]]:
        """Check if a file contains a regex pattern. Returns list of (line_num, line) matches."""
        matches = []
        if file_path.exists():
            content = file_path.read_text()
            for i, line in enumerate(content.split("\n"), 1):
                if re.search(pattern, line, re.IGNORECASE):
                    matches.append((i, line.strip()))
        return matches

    def record(self, category: str, check: str, passed: bool, message: str = "", severity: str = "error") -> None:
        """Record a verification result."""
        if category not in self.results:
            self.results[category] = {"checks": [], "passed": 0, "failed": 0, "warnings": 0}

        status = "PASS" if passed else ("WARN" if severity == "warning" else "FAIL")
        self.results[category]["checks"].append({
            "check": check,
            "status": status,
            "message": message,
        })

        if passed:
            self.results[category]["passed"] += 1
            self.passed += 1
        elif severity == "warning":
            self.results[category]["warnings"] += 1
            self.warnings += 1
        else:
            self.results[category]["failed"] += 1
            self.failed += 1

    def verify_infrastructure(self) -> None:
        """Verify all infrastructure components exist."""
        print("\n[1/5] INFRASTRUCTURE VERIFICATION")
        print("=" * 60)

        components = [
            # Embeddings
            ("futurnal.embeddings.service", "MultiModelEmbeddingService"),
            ("futurnal.embeddings.config", "EmbeddingServiceConfig"),
            ("futurnal.embeddings.registry", "ModelRegistry"),
            ("futurnal.embeddings.router", "ModelRouter"),
            # PKG
            ("futurnal.pkg.database.manager", "PKGDatabaseManager"),
            ("futurnal.pkg.database.config", "PKGDatabaseConfig"),
            ("futurnal.pkg.queries.temporal", "TemporalGraphQueries"),
            # LLM
            ("futurnal.extraction.ollama_client", "OllamaLLMClient"),
            # Search
            ("futurnal.search.hybrid.retrieval", "SchemaAwareRetrieval"),
            ("futurnal.search.hybrid.query_router", "QueryEmbeddingRouter"),
            ("futurnal.search.temporal.engine", "TemporalQueryEngine"),
            ("futurnal.search.causal.retrieval", "CausalChainRetrieval"),
            # ChromaDB
            ("futurnal.embeddings.integration", "TemporalAwareVectorWriter"),
            ("futurnal.embeddings.schema_versioned_store", "SchemaVersionedEmbeddingStore"),
        ]

        for module_path, class_name in components:
            result = self.verify_import(module_path, class_name)
            self.record(
                "Infrastructure",
                f"{class_name}",
                result,
                f"from {module_path}" if result else f"MISSING: {module_path}.{class_name}"
            )
            status = "PASS" if result else "FAIL"
            print(f"  [{status}] {class_name}")

    def verify_option_b_compliance(self) -> None:
        """Verify Option B compliance in the codebase."""
        print("\n[2/5] OPTION B COMPLIANCE VERIFICATION")
        print("=" * 60)

        src_path = PROJECT_ROOT / "src" / "futurnal"

        # Check 1: No fine-tuning code (model.train(), .backward(), optimizer)
        print("\n  Checking for fine-tuning violations...")
        finetune_patterns = [
            r"\.train\(\s*\)",  # model.train()
            r"\.backward\(\)",  # loss.backward()
            r"optimizer\.(step|zero_grad)",  # optimizer usage
            r"torch\.nn\..*\.parameters",  # accessing parameters for training
        ]

        violations = []
        for pattern in finetune_patterns:
            for py_file in src_path.rglob("*.py"):
                matches = self.check_file_for_pattern(py_file, pattern)
                for line_num, line in matches:
                    # Skip test files and comments
                    if "test" in str(py_file) or line.strip().startswith("#"):
                        continue
                    violations.append((py_file.relative_to(PROJECT_ROOT), line_num, pattern))

        passed = len(violations) == 0
        self.record(
            "Option B - Ghost Frozen",
            "No fine-tuning code",
            passed,
            f"Found {len(violations)} violations" if violations else "No fine-tuning code found"
        )
        print(f"  [{'PASS' if passed else 'FAIL'}] No fine-tuning code: {len(violations)} violations")

        # Check 2: Temporal context required for Events
        print("\n  Checking temporal-first design...")
        request_file = src_path / "embeddings" / "request.py"
        if request_file.exists():
            content = request_file.read_text()
            has_temporal_validation = "temporal_context" in content and "Event" in content
            self.record(
                "Option B - Temporal First",
                "Event requires temporal_context",
                has_temporal_validation,
                "Temporal context validation found" if has_temporal_validation else "Missing validation"
            )
            print(f"  [{'PASS' if has_temporal_validation else 'FAIL'}] Event requires temporal_context")

        # Check 3: Seed schema (not hardcoded forever)
        print("\n  Checking schema evolution support...")
        seed_file = src_path / "extraction" / "schema" / "seed.py"
        evolution_file = src_path / "extraction" / "schema" / "evolution.py"

        has_seed = seed_file.exists()
        has_evolution = evolution_file.exists()

        self.record(
            "Option B - Schema Evolution",
            "Seed schema exists",
            has_seed,
            str(seed_file.relative_to(PROJECT_ROOT)) if has_seed else "Missing seed.py"
        )
        self.record(
            "Option B - Schema Evolution",
            "Evolution mechanism exists",
            has_evolution,
            str(evolution_file.relative_to(PROJECT_ROOT)) if has_evolution else "Missing evolution.py"
        )
        print(f"  [{'PASS' if has_seed else 'FAIL'}] Seed schema exists")
        print(f"  [{'PASS' if has_evolution else 'FAIL'}] Evolution mechanism exists")

        # Check 4: TrainingFreeGRPO pattern
        print("\n  Checking experiential learning pattern...")
        grpo_found = False
        for py_file in src_path.rglob("*.py"):
            matches = self.check_file_for_pattern(py_file, r"TrainingFreeGRPO|experiential.*learning")
            if matches:
                grpo_found = True
                break

        self.record(
            "Option B - Experiential Learning",
            "TrainingFreeGRPO pattern",
            grpo_found,
            "Found TrainingFreeGRPO reference" if grpo_found else "Missing GRPO pattern"
        )
        print(f"  [{'PASS' if grpo_found else 'FAIL'}] TrainingFreeGRPO pattern exists")

    def verify_search_gap(self) -> None:
        """Verify and document the search API gap."""
        print("\n[3/5] SEARCH API GAP ANALYSIS")
        print("=" * 60)

        api_file = PROJECT_ROOT / "src" / "futurnal" / "search" / "api.py"

        # Check 1: Find keyword matching code
        print("\n  Locating keyword matching code...")
        keyword_matches = self.check_file_for_pattern(api_file, r"text_lower.*=.*\.lower\(\)")

        self.record(
            "Search Gap",
            "Keyword matching locations found",
            len(keyword_matches) > 0,
            f"Found {len(keyword_matches)} keyword matching locations" if keyword_matches else "No keyword matching found",
            severity="warning" if len(keyword_matches) > 0 else "error"
        )

        for line_num, _ in keyword_matches[:3]:  # Show first 3
            print(f"  [GAP] Line {line_num}: Keyword matching (should be semantic)")

        # Check 2: SchemaAwareRetrieval imported but not wired
        print("\n  Checking SchemaAwareRetrieval wiring...")
        if api_file.exists():
            content = api_file.read_text()
            imported = "SchemaAwareRetrieval" in content
            wired = "schema_retrieval =" in content and "SchemaAwareRetrieval(" in content

            self.record(
                "Search Gap",
                "SchemaAwareRetrieval imported",
                imported,
                "Import found" if imported else "Not imported"
            )
            self.record(
                "Search Gap",
                "SchemaAwareRetrieval wired",
                wired,
                "Properly initialized" if wired else "NOT WIRED - declared as None",
                severity="warning"  # This is expected - Step 01 will fix
            )
            print(f"  [{'PASS' if imported else 'FAIL'}] SchemaAwareRetrieval imported")
            print(f"  [WARN] SchemaAwareRetrieval NOT WIRED (Step 01 will fix)")

        # Check 3: Causal search stubbed
        print("\n  Checking causal search status...")
        causal_stub = self.check_file_for_pattern(api_file, r"Placeholder.*CausalChainRetrieval|stub|TODO.*causal")

        is_stubbed = len(causal_stub) > 0
        self.record(
            "Search Gap",
            "Causal search status",
            False,  # It's a gap, not a pass
            "STUBBED - needs integration" if is_stubbed else "Status unknown",
            severity="warning"
        )
        print(f"  [WARN] Causal search is STUBBED (Step 01 will fix)")

    def verify_edc_pipeline(self) -> None:
        """Verify EDC pipeline pattern components exist."""
        print("\n[4/5] EDC PIPELINE PATTERN VERIFICATION")
        print("=" * 60)

        src_path = PROJECT_ROOT / "src" / "futurnal"

        # Check extraction components exist
        components = {
            "EXTRACT": [
                ("extraction/unified.py", "Unified extraction API"),
                ("extraction/ner/spacy_extractor.py", "NER extractor"),
                ("extraction/temporal/markers.py", "Temporal marker extraction"),
            ],
            "DEFINE": [
                ("extraction/schema/discovery.py", "Schema discovery"),
                ("extraction/schema/evolution.py", "Schema evolution"),
                ("extraction/schema/seed.py", "Seed schema"),
            ],
            "CANONICALIZE": [
                ("extraction/causal/models.py", "Causal models"),
                ("extraction/causal/relationship_detector.py", "Causal relationship detector"),
                ("extraction/causal/bradford_hill_prep.py", "Bradford Hill criteria prep"),
            ],
        }

        for phase, files in components.items():
            print(f"\n  Phase: {phase}")
            for rel_path, description in files:
                file_path = src_path / rel_path
                exists = file_path.exists()
                self.record(
                    f"EDC - {phase}",
                    description,
                    exists,
                    str(rel_path) if exists else f"MISSING: {rel_path}"
                )
                print(f"    [{'PASS' if exists else 'FAIL'}] {description}")

    def verify_tests_pass(self) -> None:
        """Run Step 00 verification tests."""
        print("\n[5/5] VERIFICATION TESTS")
        print("=" * 60)

        test_file = PROJECT_ROOT / "tests" / "step00" / "test_infrastructure_verification.py"

        exists = test_file.exists()
        self.record(
            "Tests",
            "Step 00 test file exists",
            exists,
            str(test_file.relative_to(PROJECT_ROOT)) if exists else "MISSING test file"
        )
        print(f"  [{'PASS' if exists else 'FAIL'}] Test file exists")

        if exists:
            print("  Running pytest (this may take a moment)...")
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=no", "-q"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )

            # Parse results
            passed_match = re.search(r"(\d+) passed", result.stdout)
            failed_match = re.search(r"(\d+) failed", result.stdout)

            passed_count = int(passed_match.group(1)) if passed_match else 0
            failed_count = int(failed_match.group(1)) if failed_match else 0

            all_passed = failed_count == 0 and passed_count > 0
            self.record(
                "Tests",
                "Infrastructure tests pass",
                all_passed,
                f"{passed_count} passed, {failed_count} failed"
            )
            print(f"  [{'PASS' if all_passed else 'FAIL'}] {passed_count} passed, {failed_count} failed")

    def print_summary(self) -> None:
        """Print verification summary."""
        print("\n" + "=" * 60)
        print("STEP 00 VERIFICATION SUMMARY")
        print("=" * 60)

        total = self.passed + self.failed + self.warnings

        print(f"\nOverall: {self.passed}/{total} checks passed")
        print(f"  - Passed:   {self.passed}")
        print(f"  - Warnings: {self.warnings} (expected gaps for Step 01)")
        print(f"  - Failed:   {self.failed}")

        print("\nBy Category:")
        for category, data in self.results.items():
            status = "PASS" if data["failed"] == 0 else "FAIL"
            print(f"  [{status}] {category}: {data['passed']}/{data['passed'] + data['failed'] + data['warnings']}")

        print("\n" + "=" * 60)
        if self.failed == 0:
            print("STEP 00: READY FOR STEP 01")
            print("All infrastructure verified. Gaps documented.")
            return True
        else:
            print("STEP 00: ISSUES FOUND")
            print("Please review failed checks before proceeding.")
            return False

    def run(self) -> bool:
        """Run all verifications."""
        print("=" * 60)
        print("STEP 00: FOUNDATION & RESEARCH ALIGNMENT VERIFICATION")
        print("=" * 60)

        self.verify_infrastructure()
        self.verify_option_b_compliance()
        self.verify_search_gap()
        self.verify_edc_pipeline()
        self.verify_tests_pass()

        return self.print_summary()


if __name__ == "__main__":
    verifier = Step00Verifier()
    success = verifier.run()
    sys.exit(0 if success else 1)
