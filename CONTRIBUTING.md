# Contributing to qmri

Thank you for your interest in contributing to qmri!

## Quick Start

```bash
# Clone and set up
git clone https://github.com/gold-standard-phantoms/qmri.git
cd qmri
uv sync

# Run checks
uv run ruff check .
uv run mypy packages/qmri/src
uv run pytest
```

## Documentation

For full contributor documentation, see:

- [Development Setup](https://gold-standard-phantoms.github.io/qmri/contributing/setup/)
- [Code Style Guide](https://gold-standard-phantoms.github.io/qmri/contributing/style-guide/)
- [Testing Guide](https://gold-standard-phantoms.github.io/qmri/contributing/testing/)
- [Adding New Modules](https://gold-standard-phantoms.github.io/qmri/contributing/adding-modules/)
- [Architecture Overview](https://gold-standard-phantoms.github.io/qmri/contributing/architecture/)

## Key Points

- **Python 3.10+** required
- **UK English** spelling throughout (colour, centre, behaviour, etc.)
- **Type annotations** on all public functions
- **Google-style docstrings** with examples
- **Tests required** for new functionality

## Questions?

Open a [GitHub issue](https://github.com/gold-standard-phantoms/qmri/issues) for discussion.
