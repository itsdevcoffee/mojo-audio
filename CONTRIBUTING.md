# Contributing to mojo-audio

Thank you for your interest in contributing to mojo-audio! This document provides guidelines and instructions for contributing.

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/itsdevcoffee/mojo-audio.git
cd mojo-audio
pixi install

# Run tests
pixi run test

# Run benchmarks
pixi run bench-compare
```

---

## 📋 Ways to Contribute

We welcome contributions in these areas:

### 1. Performance Optimizations
- Further FFT improvements
- SIMD vectorization enhancements
- Memory optimization
- Platform-specific tuning (ARM, x86)

See `docs/project/01-15-2026-optimization-opportunities.md` for current optimization targets.

### 2. New Features
- Additional audio features (MFCC, CQT, etc.)
- More window functions
- Additional normalization methods
- Streaming/chunked processing

### 3. FFI Bindings
- Additional language examples (Go, Julia, Zig, Swift)
- Higher-level wrappers
- Package integrations

### 4. Documentation
- Tutorials and guides
- Algorithm explanations
- Performance analysis
- API documentation improvements

### 5. Bug Fixes
- Numerical accuracy issues
- Edge case handling
- Memory leaks
- Platform compatibility

---

## 🔧 Development Workflow

### Environment Setup

**Requirements:**
- Mojo 0.26.1 or later
- pixi package manager

**Install dependencies:**
```bash
pixi install
```

### Running Tests

```bash
# All tests
pixi run test

# Individual test suites
pixi run test-window
pixi run test-fft
pixi run test-mel
```

**Test requirements:**
- All tests must pass before submitting PR
- Add tests for new features
- Maintain or improve code coverage

### Running Benchmarks

```bash
# Full comparison (recommended)
pixi run bench-compare

# Stable benchmark (5 runs, median)
pixi run bench-stable 5

# Mojo only (optimized)
pixi run bench-optimized

# Python baseline
pixi run bench-python
```

**Benchmark guidelines:**
- Run benchmarks before and after changes
- Document performance impact in PR
- Use stable benchmark for publication-quality results

### Code Style

**Mojo conventions:**
- Use `fn` for functions (not `def`)
- Type annotations for all function parameters
- Docstrings for public functions
- `@parameter` for compile-time constants
- SIMD width detection: `simd_width_of[DType.float32]()`

**Example:**
```mojo
fn apply_window(signal: List[Float32], window: List[Float32]) raises -> List[Float32]:
    """
    Apply window function to signal.

    Args:
        signal: Input signal
        window: Window coefficients

    Returns:
        Windowed signal
    """
    # Implementation...
```

### Commit Messages

Use conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

**Examples:**
```
feat(fft): add radix-8 butterfly implementation
fix(mel): correct triangular filter edge case
perf(simd): optimize window application with AVX-512
docs(ffi): add Go usage example
```

---

## 📝 Pull Request Process

### Before Submitting

1. **Run tests:** `pixi run test`
2. **Run benchmarks:** `pixi run bench-compare`
3. **Update documentation** if needed
4. **Add tests** for new features
5. **Check code style**

### PR Guidelines

**Title:** Use conventional commit format
```
feat(component): brief description
```

**Description should include:**
- What changed and why
- Performance impact (if applicable)
- Breaking changes (if any)
- Related issues

**Example:**
```markdown
## Summary
Implements radix-8 FFT butterflies for sizes divisible by 8.

## Performance
- 1.15x faster on 512-point FFT
- Negligible impact on other sizes

## Testing
- Added radix-8 test cases
- All existing tests pass

## Related Issues
Closes #42
```

### Review Process

1. Maintainers will review within 3-5 days
2. Address review feedback
3. CI must pass (tests + benchmarks)
4. Minimum 1 approval required
5. Squash and merge

---

## 🧪 Testing Guidelines

### Test Structure

Tests are in `tests/` directory:
- `test_window.mojo` - Window functions
- `test_fft.mojo` - FFT operations
- `test_mel.mojo` - Mel filterbank

### Writing Tests

```mojo
fn test_new_feature() raises:
    """Test description."""
    # Setup
    var input = List[Float32]()

    # Execute
    var result = new_feature(input)

    # Assert
    assert len(result) == expected_length, "Length mismatch"
    assert abs(result[0] - expected_value) < 1e-6, "Value mismatch"
```

### Test Requirements

- Test edge cases (empty input, size 1, power of 2, non-power of 2)
- Test numerical accuracy (compare with reference implementation)
- Test error handling (invalid inputs)
- Test performance characteristics (no unexpected slowdowns)

---

## 📊 Benchmarking Guidelines

### Running Benchmarks

Always use improved methodology:
- Deterministic chirp signal (reproducible)
- 5 warmup runs (JIT stabilization)
- 20 iterations (statistical significance)
- Outlier exclusion (robust statistics)

### Reporting Results

**Format:**
```
Duration: 30 seconds
Mean: 27.4 ± 3.1 ms
Throughput: 1096x realtime
Comparison: 1.1x faster than librosa
```

### Performance Requirements

- No regressions on existing benchmarks
- Document speedups in PR description
- Use `pixi run bench-stable` for final measurements

---

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues
2. Verify with latest version
3. Create minimal reproduction

### Bug Report Template

```markdown
**Description:**
Brief description of the bug

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Mojo version: [e.g., 0.26.1]
- Hardware: [e.g., x86_64, ARM]

**To Reproduce:**
Steps to reproduce the behavior

**Expected behavior:**
What you expected to happen

**Actual behavior:**
What actually happened

**Code sample:**
```mojo
// Minimal code to reproduce
```

**Additional context:**
Any other relevant information
```

---

## 💡 Feature Requests

### Proposal Format

1. **Use case:** What problem does it solve?
2. **Proposed solution:** How should it work?
3. **Alternatives:** Other approaches considered
4. **Impact:** Performance/complexity trade-offs

---

## 📚 Documentation Standards

### Code Documentation

- Public functions must have docstrings
- Include Args and Returns sections
- Explain algorithm choices for complex operations
- Add inline comments for non-obvious logic

### File Documentation

- README.md for usage and examples
- docs/guides/ for tutorials
- docs/context/ for architecture decisions
- docs/research/ for technical explorations

### Documentation Style

- Use markdown format
- Include code examples
- Add performance notes where relevant
- Follow naming convention: `MM-DD-YYYY-descriptive-name.md`

---

## 🤝 Code Review

### What We Look For

- **Correctness:** Does it work as intended?
- **Performance:** Does it maintain or improve speed?
- **Code quality:** Is it readable and maintainable?
- **Tests:** Are edge cases covered?
- **Documentation:** Is it clear what changed?

### Giving Feedback

- Be respectful and constructive
- Explain the "why" behind suggestions
- Distinguish between required changes and suggestions
- Acknowledge good work

---

## 📞 Getting Help

- **Questions:** Open a GitHub Discussion
- **Bugs:** Open a GitHub Issue
- **Chat:** Join our Discord (link in README)

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to mojo-audio!** 🎵⚡
