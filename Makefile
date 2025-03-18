.PHONY: setup install clean test lint docs init classify add-docs

# Variables
PYTHON := python
PIP := pip
PROJECT := document_relevance

# Installation
install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

# Setup
setup:
	./setup.sh

# Clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Run tests
test:
	pytest tests/

# Lint code
lint:
	flake8 $(PROJECT)
	mypy $(PROJECT)

# Build documentation
docs:
	cd docs && make html

# Initialize the system with reference documents (recursive)
init:
	@echo "Initializing system with reference documents (recursive scan)..."
	@mkdir -p reference_documents
	@if [ -z "$$(ls -A reference_documents)" ]; then \
		echo "Warning: No documents found in reference_documents"; \
	else \
		$(PYTHON) -m $(PROJECT).cli init --reference-dir reference_documents --kb-path data/knowledge_base/kb.pkl; \
	fi

# Classify documents
classify:
	@echo "Classifying documents..."
	@mkdir -p new_documents
	@mkdir -p data/results
	@if [ -z "$$(ls -A new_documents)" ]; then \
		echo "Warning: No documents found in new_documents"; \
	else \
		$(PYTHON) -m $(PROJECT).cli classify --input new_documents --output data/results/results.json; \
	fi

# Add more reference documents (recursive)
add-docs:
	@echo "Adding reference documents recursively..."
	@mkdir -p reference_documents
	@if [ -z "$$(ls -A reference_documents)" ]; then \
		echo "Warning: No documents found in reference_documents"; \
	else \
		$(PYTHON) -m $(PROJECT).cli add --reference-dir reference_documents; \
	fi

# Run classification on example
run-example:
	$(PYTHON) -m $(PROJECT).cli classify --input ./new_documents --output ./data/results.json
