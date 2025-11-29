# Contributing to EPL Betting Model

First off, thanks for taking the time to contribute! 🎉

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs actual behavior
- **Environment details** (Python version, OS, etc.)
- **Code samples** if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Use case** for the feature
- **Proposed solution** with examples
- **Alternative solutions** you've considered

### Pull Requests

1. Fork the repo and create your branch from `main`
2. Install development dependencies: `pip install -e ".[dev]"`
3. Make your changes
4. Add tests for new functionality
5. Ensure tests pass: `pytest tests/ -v`
6. Format code: `black src/ tests/`
7. Lint code: `flake8 src/ tests/`
8. Update documentation if needed
9. Submit your PR!

## Code Style

- Follow [PEP 8](https://pep8.org/) guidelines
- Use [Black](https://github.com/psf/black) for formatting
- Write docstrings for all public functions (Google style)
- Add type hints for function parameters and returns
- Keep functions focused and under 50 lines when possible

## Testing

- Write tests for new features
- Maintain >80% code coverage
- Use pytest fixtures for common setup
- Name tests descriptively: `test_<function>_<scenario>`

## Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Keep first line under 72 characters
- Reference issues when applicable

Example:
```
Add Kelly criterion variants for risk analysis

- Implement full, half, and quarter Kelly
- Add unit tests for edge cases
- Update documentation

Closes #42
```

## Project Structure

```
src/
├── predictor.py      # Core prediction model
├── backtest.py       # Backtesting engine
├── risk_analysis.py  # Risk metrics
└── utils.py          # Helper functions
```

## Questions?

Feel free to open an issue with the "question" label!
