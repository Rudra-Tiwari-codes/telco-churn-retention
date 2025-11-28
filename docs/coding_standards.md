# Coding Standards

This document outlines the coding standards and conventions for the Telco Churn Retention Platform.

## Code Style

- **Python Version**: Python 3.11+
- **Line Length**: 100 characters
- **Formatter**: Black (configured in `pyproject.toml`)
- **Linter**: Ruff (configured in `pyproject.toml`)
- **Type Checking**: mypy (configured in `pyproject.toml`)

## Code Formatting

All code must be formatted with Black before committing:

```bash
black src/ tests/ scripts/
```

## Linting

Run Ruff to check for code quality issues:

```bash
ruff check src/ tests/ scripts/
```

Auto-fix issues where possible:

```bash
ruff check --fix src/ tests/ scripts/
```

## Type Hints

- Use type hints for all function signatures
- Use `from __future__ import annotations` for forward references
- Use `typing` module for complex types (e.g., `Dict`, `List`, `Optional`)
- Use `pathlib.Path` instead of strings for file paths

Example:

```python
from __future__ import annotations
from pathlib import Path
from typing import Optional

def process_data(input_path: Path, output_path: Optional[Path] = None) -> pd.DataFrame:
    ...
```

## Naming Conventions

- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Module names**: `snake_case`

## Documentation

- Use docstrings for all public functions, classes, and modules
- Follow Google-style docstrings
- Include type information in docstrings when helpful

Example:

```python
def load_dataset(path: Path) -> pd.DataFrame:
    """Load the Telco churn dataset from a CSV file.
    
    Args:
        path: Path to the CSV file.
        
    Returns:
        DataFrame containing the loaded data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    ...
```

## Testing

- Write tests for all public functions and classes
- Use pytest for testing
- Test files should be in `tests/` directory
- Test file names should start with `test_`
- Aim for >80% code coverage

## Imports

- Group imports: standard library, third-party, local
- Use absolute imports for project code
- Sort imports with Ruff (isort)

Example:

```python
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.data.ingestion import load_dataset
```

## Error Handling

- Use specific exception types
- Include helpful error messages
- Log errors appropriately

## Git Workflow

- Use meaningful commit messages in imperative mood
- Branch naming: `feat/<scope>`, `fix/<scope>`, `chore/<scope>`
- All code must pass linting, formatting, type checking, and tests before merging
- Use pre-commit hooks to enforce standards

## Pre-commit Hooks

Pre-commit hooks are configured to automatically:
- Format code with Black
- Lint code with Ruff
- Check types with mypy
- Run tests

Install and run pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

