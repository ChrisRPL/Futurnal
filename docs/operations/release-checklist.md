# Release Checklist

**Futurnal v1.0.0 - Phase 1 (Archivist)**

Use this checklist for every production release.

---

## Pre-Release (1-2 days before)

### Code Quality
- [ ] All PRs merged and reviewed
- [ ] No critical or high-priority bugs open
- [ ] Code freeze applied to release branch
- [ ] Version number updated in all files:
  - [ ] `pyproject.toml`
  - [ ] `desktop/package.json`
  - [ ] `desktop/src-tauri/tauri.conf.json`

### Testing
- [ ] Full test suite passes: `pytest tests/ -v`
- [ ] Quality gate tests pass: `pytest tests/quality_gates/ -v`
- [ ] Security tests pass: `pytest tests/security/ -v`
- [ ] Performance tests pass: `pytest tests/performance/ -v`
- [ ] Manual smoke tests completed on all platforms

### Quality Gates (from Step 09)
- [ ] Temporal accuracy > 85%
- [ ] Schema alignment > 90%
- [ ] Extraction precision >= 0.8
- [ ] Ghost model verified FROZEN
- [ ] Causal ordering 100% valid
- [ ] Token priors natural language only

### Documentation
- [ ] CHANGELOG.md updated with release notes
- [ ] User documentation reviewed
- [ ] API documentation current
- [ ] Privacy policy reviewed

---

## Build Phase

### Python Package
- [ ] Clean build environment: `rm -rf dist/ build/ *.egg-info`
- [ ] Build package: `python -m build`
- [ ] Verify package: `twine check dist/*`
- [ ] Test local install: `pip install dist/*.whl`

### Desktop Application

#### macOS
- [ ] Build ARM64: `npm run tauri build -- --target aarch64-apple-darwin`
- [ ] Build x64: `npm run tauri build -- --target x86_64-apple-darwin`
- [ ] Sign builds with Developer ID
- [ ] Notarize with Apple
- [ ] Test .dmg installation
- [ ] Test app launch and basic functionality

#### Windows
- [ ] Build x64: `npm run tauri build -- --target x86_64-pc-windows-msvc`
- [ ] Sign with code signing certificate
- [ ] Test .msi installation
- [ ] Test app launch and basic functionality

#### Linux
- [ ] Build x64: `npm run tauri build -- --target x86_64-unknown-linux-gnu`
- [ ] Test .AppImage on Ubuntu 22.04
- [ ] Test .deb installation
- [ ] Test app launch and basic functionality

### Artifacts
- [ ] Generate checksums for all artifacts
- [ ] Verify checksums match builds
- [ ] Upload to release staging area

---

## Release Phase

### GitHub Release
- [ ] Create draft release on GitHub
- [ ] Upload all build artifacts:
  - [ ] macOS ARM64 (.dmg)
  - [ ] macOS x64 (.dmg)
  - [ ] Windows x64 (.msi)
  - [ ] Linux x64 (.AppImage, .deb)
  - [ ] Python package (.tar.gz, .whl)
  - [ ] checksums.txt
- [ ] Write release notes:
  - [ ] Summary of changes
  - [ ] Breaking changes (if any)
  - [ ] Migration instructions (if any)
  - [ ] Known issues
- [ ] Review release page
- [ ] Publish release (remove draft status)

### PyPI (if applicable)
- [ ] Upload to Test PyPI first: `twine upload --repository testpypi dist/*`
- [ ] Test install from Test PyPI
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Verify PyPI page

---

## Post-Release Validation

### Download & Install Tests
- [ ] Download installers from release page
- [ ] Verify checksums match
- [ ] Test fresh install on macOS
- [ ] Test fresh install on Windows
- [ ] Test fresh install on Linux
- [ ] Test upgrade from previous version

### Functionality Tests
- [ ] First launch / onboarding flow
- [ ] Add Obsidian vault
- [ ] Run ingestion
- [ ] Search functionality
- [ ] Chat functionality
- [ ] Graph visualization
- [ ] Settings / privacy controls

### Health Checks
- [ ] `futurnal health check` passes
- [ ] All services connecting properly
- [ ] No errors in logs

---

## Communication

### Announcements
- [ ] Update website/landing page (if applicable)
- [ ] Social media announcement (if applicable)
- [ ] Email to beta users (if applicable)

### Support Preparation
- [ ] Support team briefed on new features
- [ ] Known issues documented
- [ ] FAQ updated

---

## Monitoring (24-48 hours post-release)

### Error Tracking
- [ ] Monitor for crash reports
- [ ] Monitor for error spikes
- [ ] Review user feedback channels

### Performance
- [ ] Check telemetry dashboards (if enabled)
- [ ] Monitor latency metrics
- [ ] Check for memory issues

### Rollback Readiness
- [ ] Rollback procedure documented
- [ ] Previous version artifacts available
- [ ] Rollback tested in staging

---

## Sign-Off

### Release Manager
- Name: ____________________
- Date: ____________________
- Signature: ____________________

### Quality Lead
- Name: ____________________
- Date: ____________________
- Signature: ____________________

### Security Lead
- Name: ____________________
- Date: ____________________
- Signature: ____________________

---

## Version-Specific Notes

### v1.0.0 (Phase 1 - Archivist)

**Key Features:**
- Local-first personal knowledge engine
- Obsidian vault integration
- Intelligent search with intent detection
- Chat with knowledge context
- Graph visualization
- Privacy-preserving architecture

**Dependencies:**
- Ollama (local LLM)
- Neo4j (graph database)
- ChromaDB (vector store)

**Known Issues:**
- None at release

**Migration Notes:**
- Fresh install recommended for v1.0.0
- No migration from pre-release versions

---

*Part of Step 10: Production Readiness*
*Phase 1 (Archivist) - December 2024*
