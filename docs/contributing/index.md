# Contributing

Thank you for your interest in contributing to qmri! This section covers everything you need to get started.

## Overview

qmri is a monorepo containing multiple independently-installable packages that share a common namespace:

```
qmri/
├── packages/
│   ├── qmri/          # Core package (numpy, scipy only)
│   ├── qmri-io/       # File I/O (adds nibabel)
│   ├── qmri-cli/      # Command-line interface (adds click, rich)
│   ├── qmri-viz/      # Visualisation (adds matplotlib)
│   └── qmri-dro/      # Digital Reference Objects
├── tests/             # Test suite
├── docs/              # Documentation
└── pyproject.toml     # Workspace configuration
```

## Getting Started

1. **[Development Setup](setup.md)** — Clone, install, and configure your environment
2. **[Code Style Guide](style-guide.md)** — Linting, formatting, type hints, and docstrings
3. **[Testing Guide](testing.md)** — Running and writing tests
4. **[Adding New Modules](adding-modules.md)** — How to add new signal models or fitting algorithms
5. **[Architecture Overview](architecture.md)** — Design philosophy and package structure

## Quick Reference

### Commands

```bash
# Install dependencies
uv sync

# Run linting
uv run ruff check .
uv run ruff format --check .

# Run type checking
uv run mypy packages/qmri/src

# Run tests
uv run pytest

# Build documentation
uv run mkdocs serve
```

### Key Conventions

| Convention | Example |
|------------|---------|
| UK English | colour, centre, behaviour |
| Type hints | `def fit(signal: NDArray[np.floating]) -> ADCResult` |
| Docstrings | Google style with examples |
| Tests | pytest, property-based with Hypothesis |

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `mypy packages/qmri/src` passes
- [ ] `pytest` passes
- [ ] New code has test coverage
- [ ] Documentation updated if needed
- [ ] UK English spelling used

## Questions?

Open a [GitHub issue](https://github.com/gold-standard-phantoms/qmri/issues) for discussion.
