.PHONY: setup lint format typecheck test check precommit docker-build docker-run

setup:
	pip install -e ".[torch,jax,dev]"
	pre-commit install

lint:
	ruff check src tests

format:
	black src tests
	ruff check --fix src tests

typecheck:
	mypy src

test:
	pytest --cov=matryoshka_search --cov-report=term-missing

check: lint typecheck test

precommit:
	pre-commit run --all-files

docker-build:
	docker build -t matryoshka-search:latest .

docker-run:
	docker run --rm -it -v $(PWD)/data:/app/data matryoshka-search:latest
