# kedro_starter

Projeto-esqueleto inspirado no template oficial do Kedro.
Inclui uma pipeline simples de exemplo (+ testes), camadas de dados e configuração padrão.

## Requisitos
- Python 3.9+
- pipx (opcional) ou pip
- (Opcional) virtualenv/venv

## Início rápido
```bash
# criar venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
# instalar
pip install -U pip
pip install kedro>=0.19.0 pandas pytest
# rodar testes
pytest -q
# executar pipeline
kedro run
# (Opcional) visualizar pipeline
kedro viz
```

## Estrutura
Veja comentários inline nos arquivos e a explicação no chat.
