###############################################################################
# 📦 Makefile – Demand Forecast Pipeline
# ---------------------------------------------------------------------------
# Comandos básicos para desenvolvimento e execução do projeto Kedro
###############################################################################

PROJECT_NAME := demand_forecast_pipeline
PYTHON       ?= python3
VENV         ?= .venv

.DEFAULT_GOAL := help

## 🛠️ Criar ambiente virtual e instalar dependências
install:
	@echo "📦 Criando ambiente virtual..."
	$(PYTHON) -m venv $(VENV)
	@echo "📦 Instalando dependências..."
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

## ▶️ Executar pipeline completo
run:
	@echo "🚀 Executando pipeline Kedro..."
	$(VENV)/bin/kedro run

## ▶️ Executar pipeline por tag
run-tag:
	@read -p "Digite a tag: " tag; \
	$(VENV)/bin/kedro run --tag $$tag

## 🧪 Executar testes
test:
	@echo "🧪 Rodando testes..."
	$(VENV)/bin/pytest -q

## 🧹 Limpar arquivos temporários
clean:
	@echo "🧹 Limpando cache e arquivos temporários..."
	rm -rf .pytest_cache
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +

## 📜 Mostrar ajuda
help:
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
		sed 's/:.*##/ →/' | sort
