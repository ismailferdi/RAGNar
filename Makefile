install:
	pip install -r requirements.txt

run-api:
	uvicorn backend.main:app --reload

run-ui:
	PYTHONPATH=. streamlit run frontend/app.py

test:
	pytest tests/ -v

eval:
	python eval/run_eval.py

wipe-vector-store:
	rm -rf vector_store/* && touch vector_store/.gitkeep

ingest-eval-docs:
	for document in eval/documents/*.pdf; do \
		curl -X POST "http://localhost:8000/ingest/" -F "file=@$$document"; \
	done