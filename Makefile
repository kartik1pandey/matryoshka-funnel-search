.PHONY: setup lint format typecheck test check precommit docker-build docker-run demo demo-web

setup:
	pip install -e ".[torch,jax,dev]"
	pre-commit install

lint:
	ruff check src tests scripts app.py

format:
	black src tests scripts app.py
	ruff check --fix src tests scripts app.py

typecheck:
	mypy src scripts app.py

test:
	pytest --cov=matryoshka_search --cov-report=term-missing

check: lint typecheck test

precommit:
	pre-commit run --all-files

demo:
	python -m matryoshka_search.demo.cli

demo-web:
	python -m matryoshka_search.demo.web

docker-build:
	docker build -t matryoshka-search:latest .

docker-run:
	docker run --rm -it -v $(PWD)/data:/app/data -v $(PWD)/checkpoints:/app/checkpoints matryoshka-search:latest
