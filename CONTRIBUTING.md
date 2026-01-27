# Contribution Guidelines

Thank you for your interest in the project! We welcome contributions to Test Metal.

## How to Start?

1. **Fork** the repository
2. **Clone** your fork
3. **Create a branch** for your feature: `git checkout -b feature/amazing-feature`
4. **Commit** your changes: `git commit -m 'Add amazing feature'`
5. **Push** to your fork: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

## Code Requirements

- Use Python 3.10+
- Follow PEP 8 coding style
- Add type hints
- Write docstrings for functions and classes
- Add tests for new features

## Testing

Before submitting a Pull Request, make sure that:

```bash
# All tests pass
pytest

# No formatting issues
python -m flake8 test_metal/

# Test coverage is at an acceptable level
pytest --cov=test_metal
```

## Bug Reports

When opening an Issue, please include:
- Description of the problem
- Steps to reproduce
- Expected behavior
- Python version and dependencies

## License

By contributing, you agree to license your code under the MIT License.
