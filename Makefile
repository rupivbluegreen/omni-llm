.PHONY: serve train-tokenizer pretrain sft dpo gate decontaminate test lint format eval clean

serve:
	uvicorn server.api.main:app --host 0.0.0.0 --port 8000 --reload

train-tokenizer:
	python -m tokenizer.train_tokenizer

pretrain:
	python -m training.pretrain

sft:
	python -m training.sft

dpo:
	python -m training.dpo

gate:
	python -m evals.gate

decontaminate:
	python -m data.decontaminate

test:
	python -m pytest tests/ -v

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

eval:
	python -m evals.run_eval

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	rm -rf .pytest_cache .ruff_cache
