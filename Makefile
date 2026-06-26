.PHONY: fixtures

fixtures:
	PYTHONPATH=. uv run python tests/fixtures/gen_rating_golden.py tests/fixtures/rating_golden.json
