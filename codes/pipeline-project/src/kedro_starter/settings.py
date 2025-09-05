"""Configurações do projeto Kedro (hooks, etc.).

Você pode adicionar hooks personalizados aqui. Por ora, mantemos vazio.
"""
# Exemplos de hooks: from kedro.framework.hooks import hook_impl

from kedro_mlflow.framework.hooks import MlflowHook
HOOKS = (MlflowHook(),)

