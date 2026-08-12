# Contributing to autoredteam

Thanks for your interest in contributing. Whether it's a bug report, new attack pack, feature request, or documentation improvement — all contributions are welcome.

## Getting Started

```bash
git clone https://github.com/Glacis-io/autoredteam.git
cd autoredteam
pip install -e ".[all]"
```

## Ways to Contribute

### Report Bugs

Open an issue using the [bug report template](https://github.com/Glacis-io/autoredteam/issues/new?template=bug_report.yml). Include:

- What you ran (CLI command or code snippet)
- What you expected
- What actually happened
- Python version and OS

### Suggest Features

Open an issue using the [feature request template](https://github.com/Glacis-io/autoredteam/issues/new?template=feature_request.yml).

### Add an Attack Pack

Attack packs live in `attack_packs/domains/`. To add a new one:

1. Create `attack_packs/domains/your_domain.py`
2. Subclass `AttackPack` from `attack_packs/base.py`
3. Define categories and seed templates following the pattern in `attack_packs/domains/healthcare.py`
4. Register the pack in `attack_packs/registry.py`
5. Add tests
6. Open a PR

### Add a Provider

Provider adapters live in `providers/`. To add support for a new LLM provider:

1. Create `providers/your_provider.py`
2. Implement the provider interface from `providers/base.py`
3. Register in `providers/registry.py`
4. Add to `providers/catalog.py`
5. Open a PR

### Improve Documentation

Documentation improvements are always welcome. If something is unclear in the README or you found a gap, open a PR.

## Development Workflow

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run tests: `python -m pytest tests/`
4. Ensure your code follows existing style conventions
5. Open a pull request against `main`

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Write a clear description of what changed and why
- Link to any related issues
- Add tests for new functionality
- Update documentation if behavior changes

## Code Style

- Follow existing patterns in the codebase
- Use type hints for function signatures
- Keep modules focused — one responsibility per file

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run a specific test
python -m pytest tests/test_v03_overhaul.py -v

# Dry run to verify CLI works
autoredteam run --dry-run
```

## Security

If you discover a security vulnerability, please report it responsibly. See [SECURITY.md](SECURITY.md) for details. Do not open a public issue for security vulnerabilities.

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
