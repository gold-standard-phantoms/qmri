# Development Setup

This guide covers setting up a development environment for qmri.

## Prerequisites

- **Python 3.10 or later**
- **[uv](https://docs.astral.sh/uv/)** package manager (recommended)
- **Git**

## Installation

### 1. Fork and Clone

```bash
git clone https://github.com/gold-standard-phantoms/qmri.git
cd qmri
```

### 2. Install Dependencies

```bash
uv sync
```

This installs all packages in the workspace plus development dependencies.

### 3. Activate the Virtual Environment

```bash
source .venv/bin/activate
```

Or use `uv run` to run commands within the environment:

```bash
uv run pytest
uv run mypy packages/qmri/src
```

## Verify Installation

```bash
# Check all tools work
uv run python -c "from qmri.diffusion import adc; print('qmri OK')"
uv run ruff --version
uv run mypy --version
uv run pytest --version
```

## IDE Setup

### VS Code

Recommended extensions:

- **Python** (Microsoft)
- **Pylance** (for type checking)
- **Ruff** (for linting/formatting)

Settings (`.vscode/settings.json`):

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.analysis.typeCheckingMode": "strict",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true
    },
    "ruff.lint.run": "onSave"
}
```

### PyCharm

1. Set interpreter to `.venv/bin/python`
2. Enable Ruff plugin
3. Configure Google-style docstrings: Settings → Tools → Python Integrated Tools → Docstrings → Google

## Working with the Monorepo

Each package can be developed independently:

```bash
# Work on core package
cd packages/qmri
uv run pytest ../../tests/test_diffusion/

# Work on I/O package
cd packages/qmri-io
uv run pytest ../../tests/test_io/
```

Packages reference each other via workspace dependencies, so changes are immediately available across packages without reinstalling.

## Building Documentation

```bash
# Serve locally with hot reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

Documentation will be available at `http://127.0.0.1:8000`.

## Common Issues

### Import Errors

If you see import errors, ensure the workspace is synced:

```bash
uv sync
```

### Type Errors from Dependencies

If mypy reports errors from dependencies, they may need stubs:

```bash
uv add --dev types-jsonschema
```

### Test Discovery Issues

Ensure `pytest` is run from the repository root:

```bash
cd /path/to/qmri
uv run pytest
```
