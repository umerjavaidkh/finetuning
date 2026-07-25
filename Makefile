.PHONY: test extract dataset

VENV=.venv/bin

test:
	$(VENV)/python3 -m pytest tests/ -v

extract:
	$(VENV)/python3 scripts/prepare_dataset.py --pdf $(PDF) --grade-level $(GRADE)

dataset:
	$(VENV)/python3 scripts/prepare_dataset.py --pdf $(PDF) --grade-level $(GRADE) --generate --limit $(LIMIT)
