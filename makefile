.PHONY: run

run:
		pdm run python main.py

.PHONY: test

test:
		pdm run pytest --cov=main tests/
